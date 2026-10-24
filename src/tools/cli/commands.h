// Subcommand interface for the `moss-infer` CLI. To add a subcommand: declare it here, implement it
// in its own .cpp under src/tools/cli/, and add one row to kCommands in main.cpp.
#pragma once

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include "mossinfer/mossinfer.h"  // MOSSINFER_API
#include "util/path.h"
#include "util/regular_file.h"

namespace mi::cli {

// `args` excludes the program name and the subcommand. Returns the process exit code.
using CommandFn = int (*)(const std::vector<std::string>& args);

int cmd_index(const std::vector<std::string>& args);
int cmd_inspect(const std::vector<std::string>& args);
int cmd_generate(const std::vector<std::string>& args);
int cmd_quantize(const std::vector<std::string>& args);

// A path argument (`what`: its flag), through the boundary check (util::check_user_path: a Windows device or pipe path
// is refused before any file-system call touches it). Every path argument of the CLI enters through here, except the
// --params file (read as a stream, process substitution included).
inline std::filesystem::path path_arg(std::string_view utf8, const std::string& what) {
  std::filesystem::path p = util::path_from_utf8(utf8);
  util::check_user_path(p, what);
  return p;
}

// The whole CLI: `argv` as UTF-8 strings, program name first. Returns the process exit code. Exceptions, output and
// locale state stay inside, so the process entry (src/tools/moss_infer/main.cpp) only converts argv.
int run_main(const std::vector<std::string>& argv);

}  // namespace mi::cli

// run_main() for the process entry, which may sit in another module than the CLI: in release packages
// (MOSSINFER_CLI_SHARED) the CLI is compiled into libmossinfer and moss-infer calls this export. `argv`: UTF-8, program
// name first. Internal to moss-infer, not part of the C API, may change without notice.
extern "C" MOSSINFER_API int mi_internal_cli_main(int argc, char** argv);
