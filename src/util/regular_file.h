// Opening a user-supplied path for reading only when it names a regular file. open(2) on a FIFO blocks until a writer
// appears, and a character device may block or never end, so a model, LoRA or image path naming one would hang the
// caller; a directory would fail later with a less clear error. Every reader of such paths opens through here:
// weights::FileReader and MappedFile (safetensors files, the supported-model checks, images, hashes) and
// read_replaceable_file (caches, the model index).
#pragma once

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
// converter's input and output, the pipeline's options and request, image and PNG paths) before any std::filesystem
// call touches it: MSVC's canonical, weakly_canonical and equivalent (and status,
// exists, file_size on a reparse point) open the path without SECURITY_SQOS_PRESENT. On Windows it throws
// std::invalid_argument "<what>: not a regular file: <path>" when is_windows_device_path says so; elsewhere it does
// nothing (POSIX device files are refused where they are opened, open_regular_file).
void check_user_path(const std::filesystem::path& path, std::string_view what);

// A path's identity for cache keys, without opening it on Windows: std::filesystem::weakly_canonical on POSIX (as
// before); on Windows std::filesystem::absolute (GetFullPathNameW, which opens nothing), lexically normalized.
std::filesystem::path path_key(const std::filesystem::path& path);

#ifdef _WIN32
// windows_path_names_device of the path or of its GetFullPathNameW form (true too when there is no full path): such a
// path is never handed to CreateFileW. (A pipe opened without SECURITY_SQOS_PRESENT would let its server impersonate
// the caller; a device's open runs its driver.) Every CreateFileW of a user path checks it first.
bool is_windows_device_path(const std::filesystem::path& path);

// kNotRegular without opening anything for is_windows_device_path. Then a probe: CreateFileW(FILE_READ_ATTRIBUTES,
// share-all, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS | SECURITY_SQOS_PRESENT | SECURITY_IDENTIFICATION), which reads
// no data and runs no device read; its handle must be FILE_TYPE_DISK with neither FILE_ATTRIBUTE_DIRECTORY nor
// FILE_ATTRIBUTE_DEVICE in its FileAttributeTagInfo. Then CreateFileW(GENERIC_READ, share-all, OPEN_EXISTING, `flags`
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

}  // namespace mi::util
