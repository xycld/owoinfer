// moss-infer's process entry: converts argv to UTF-8 and runs the CLI (src/tools/cli/), which is linked in statically
// or, in release packages (MOSSINFER_CLI_SHARED), lives in libmossinfer. Kept out of src/tools/cli/ so that the CLI
// sources compiled into the library never contain a main().
#include "tools/cli/commands.h"

#ifdef _WIN32
#include <string>
#include <utility>
#include <vector>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

// char** argv is in the ANSI code page on Windows and loses characters outside it; take the
// UTF-16 command line and convert, so commands see UTF-8 like on Linux.
int wmain(int argc, wchar_t** argv) {
  std::vector<std::string> utf8;
  for (int i = 0; i < argc; ++i) {
    const int n = WideCharToMultiByte(CP_UTF8, 0, argv[i], -1, nullptr, 0, nullptr, nullptr);
    std::string s(n > 0 ? static_cast<std::size_t>(n) : 1, '\0');
    if (n > 0) WideCharToMultiByte(CP_UTF8, 0, argv[i], -1, s.data(), n, nullptr, nullptr);
    s.pop_back();  // the converted terminator
    utf8.push_back(std::move(s));
  }
  std::vector<char*> ptrs;
  for (auto& s : utf8) ptrs.push_back(s.data());
  ptrs.push_back(nullptr);
  return mi_internal_cli_main(argc, ptrs.data());
}
#else
int main(int argc, char** argv) { return mi_internal_cli_main(argc, argv); }
#endif
