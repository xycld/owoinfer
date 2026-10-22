#include <exception>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "tools/cli/commands.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

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
    {"quantize", mi::cli::cmd_quantize, "quantize <dit> -o <out.safetensors> [--fuse-qkv] [--keep-bf16 <globs>] [--lora <f> [--lora-scale s]]   INT8 ConvRot DiT"},
};

void usage(std::ostream& os) {
  os << "usage: moss-infer <command> [args]\n\ncommands:\n";
  for (const auto& c : kCommands) os << "  " << c.usage << "\n";
}

// `argv` as UTF-8 strings, program name first.
int run(const std::vector<std::string>& argv) {
  const std::size_t argc = argv.size();
  if (argc < 2 || argv[1] == "-h" || argv[1] == "--help") {
    usage(argc < 2 ? std::cerr : std::cout);
    return argc < 2 ? 2 : 0;
  }
  const std::string& name = argv[1];
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

}  // namespace

#ifdef _WIN32
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
  return run(utf8);
}
#else
int main(int argc, char** argv) { return run(std::vector<std::string>(argv, argv + argc)); }
#endif
