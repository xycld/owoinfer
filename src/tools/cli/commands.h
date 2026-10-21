// Subcommand interface for the `moss-infer` CLI. To add a subcommand: declare it here, implement it
// in its own .cpp under src/tools/cli/, and add one row to kCommands in main.cpp.
#pragma once

#include <string>
#include <vector>

namespace mi::cli {

// `args` excludes the program name and the subcommand. Returns the process exit code.
using CommandFn = int (*)(const std::vector<std::string>& args);

int cmd_index(const std::vector<std::string>& args);
int cmd_inspect(const std::vector<std::string>& args);
int cmd_generate(const std::vector<std::string>& args);

}  // namespace mi::cli
