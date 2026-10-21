// UTF-8 <-> std::filesystem::path. path::string() and path(std::string) use the ANSI code page on
// Windows, which mangles (or throws on) non-ASCII names; these go through char8_t instead. On POSIX
// both are byte-for-byte copies.
#pragma once

#include <filesystem>
#include <string>
#include <string_view>

namespace mi::util {

// Native path as UTF-8, for messages and for strings persisted to disk.
inline std::string path_utf8(const std::filesystem::path& p) {
  const std::u8string s = p.u8string();
  return std::string(s.begin(), s.end());
}

inline std::filesystem::path path_from_utf8(std::string_view s) {
  return std::filesystem::path(std::u8string(s.begin(), s.end()));
}

}  // namespace mi::util
