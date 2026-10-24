// Whole-file reads of the small files the engine replaces atomically while others may read them (rename_replace.h):
// the hardware profile with the fused self-check, quality certificates and their hash cache, the embedding cache's
// entries and the model index. A reader must not mistake a replace in flight for a missing or broken file (a planner
// that sees "no profile" measures again and may overwrite a learned one).
//
// POSIX: one plain read (a rename never disturbs an open or an opening reader). Windows: the reader opens with
// FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, so it never blocks a writer's rename over the file, and an
// open that fails with ERROR_SHARING_VIOLATION or ERROR_ACCESS_DENIED (what a MoveFileEx / ReplaceFile in flight, or a
// file pending deletion, gives an opener for a moment) is retried with a short backoff, about 100 ms in all. A missing
// file is reported at once. The trade-off of FILE_SHARE_WRITE: a writer modifying the file in place during the read
// could hand this reader torn bytes; the engine's writers never do (they rename a finished temporary file), and the
// readers validate what they read (JSON parse, checksums), so a torn read is rejected rather than used.
#pragma once

#include <cstddef>
#include <filesystem>
#include <string>

namespace mi::util {

enum class ReadStatus {
  kOk,
  kMissing,   // no such file (or a path component is missing)
  kTooLarge,  // more than max_bytes (`out` holds nothing)
  kError,     // could not be opened or read (still locked after the retries, a permission, an I/O error, or not a
              // regular file: a FIFO or a device is refused at once, util::open_regular_file)
};

// The whole of `path` into `out` (cleared first) when it holds at most `max_bytes`.
ReadStatus read_replaceable_file(const std::filesystem::path& path, std::size_t max_bytes, std::string& out);

}  // namespace mi::util
