// Opening a user-supplied path for reading only when it names a regular file. open(2) on a FIFO blocks until a writer
// appears, and a character device may block or never end, so a model, LoRA, image or certificate path naming one would
// hang the caller; a directory would fail later with a less clear error. Every reader of such paths opens through here:
// weights::FileReader and MappedFile (safetensors files, the supported-model checks, images, hashes) and
// read_replaceable_file (certificates, caches, the model index).
#pragma once

#include <filesystem>

namespace mi::util {

enum class OpenStatus {
  kOk,
  kNotRegular,  // a FIFO, a device, a socket or a directory: closed again before any read
  kError,       // the open (or fstat) failed: errno (POSIX) / GetLastError() (Windows) holds the cause
};

// The message for kNotRegular, after the path: "<path>: not a regular file".
inline constexpr const char* kNotRegularFile = "not a regular file";

#ifdef _WIN32
// kNotRegular without opening anything for a device-namespace path (\\.\...) or a path whose attributes say directory
// or device; then CreateFileW(GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, OPEN_EXISTING,
// `flags` as dwFlagsAndAttributes), and the handle must be FILE_TYPE_DISK with neither FILE_ATTRIBUTE_DIRECTORY nor
// FILE_ATTRIBUTE_DEVICE in its FileAttributeTagInfo (a named pipe, a console or another device is not). `handle`
// receives the HANDLE on kOk.
OpenStatus open_regular_file(const std::filesystem::path& path, unsigned long flags, void*& handle);
#else
// stat(2) first: kNotRegular without opening anything that is not a regular file (the open of a device node can act on
// the device). Then open(2) with O_RDONLY | O_CLOEXEC | O_NOCTTY | O_NONBLOCK | `flags` (e.g. O_DIRECT), so a FIFO
// swapped in since the stat cannot block the open; an EWOULDBLOCK / EAGAIN (a lease being broken) on a path that is
// still a regular file is retried without O_NONBLOCK. fstat then refuses anything but a regular file (the path may have
// been replaced meanwhile), and O_NONBLOCK is cleared again for the reads. `fd` receives the descriptor on kOk.
OpenStatus open_regular_file(const std::filesystem::path& path, int flags, int& fd);
#endif

}  // namespace mi::util
