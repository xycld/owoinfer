#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include "tools/cli/commands.h"

namespace {

struct Command {
  const char* name;
  mi::cli::CommandFn run;
  const char* usage;
};

constexpr Command kCommands[] = {
    {"index", mi::cli::cmd_index, "index <bundle_dir> [-o <path>] [--checksum]   build a .mossidx for a model bundle"},
    {"inspect", mi::cli::cmd_inspect, "inspect <bundle_dir|.mossidx>                  print a weight-group summary"},
    {"generate", mi::cli::cmd_generate, "generate (--params <json> | --prompt <text>) --model <bundle_dir> -o <out.png> [...]   text to image"},
    {"check", mi::cli::cmd_check, "check <generate's options, -o optional>   check a configuration and request without loading weights"},
    {"quantize", mi::cli::cmd_quantize, "quantize <dit> -o <out.safetensors> [--format int8|nvfp4] [--recipe r] [--keep units] [--fuse-qkv] [--keep-bf16 <globs>] [--lora <f> [--lora-scale s]]   INT8 ConvRot or NVFP4 DiT"},
};

void usage(std::ostream& os) {
  os << "usage: moss-infer [--version] <command> [args]\n\ncommands:\n";
  for (const auto& c : kCommands) os << "  " << c.usage << "\n";
}

}  // namespace

int mi::cli::run_main(const std::vector<std::string>& argv) {
  const std::size_t argc = argv.size();
  if (argc < 2 || argv[1] == "-h" || argv[1] == "--help") {
    usage(argc < 2 ? std::cerr : std::cout);
    return argc < 2 ? 2 : 0;
  }
  const std::string& name = argv[1];
  if (name == "--version") {
    // moss-server's version line, without the protocol the CLI does not speak.
    std::cout << "moss-infer " << MI_VERSION_MAJOR << "." << MI_VERSION_MINOR << "." << MI_VERSION_PATCH << " (ABI "
              << MI_ABI_VERSION << ")\n";
    return 0;
  }
  std::vector<std::string> args(argv.begin() + 2, argv.end());
  for (const auto& c : kCommands) {
    if (name != c.name) continue;
    try {
      return c.run(args);
    } catch (const std::exception& e) {
      std::cerr << "moss-infer " << name << ": " << e.what() << "\n";
      return 1;
    }
  }
  std::cerr << "moss-infer: unknown command '" << name << "'\n\n";
  usage(std::cerr);
  return 2;
}

// Declared in commands.h: internal to moss-infer, not part of the C API, may change without notice.
int mi_internal_cli_main(int argc, char** argv) {
  return mi::cli::run_main(std::vector<std::string>(argv, argv + argc));
}
