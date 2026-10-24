// `moss-infer index` and `moss-infer inspect`: build / display the weight-group index of a bundle.
#include <filesystem>
#include <iostream>
#include <stdexcept>

#include "tools/cli/commands.h"
#include "util/path.h"
#include "weights/model_index.h"
#include "weights/supported_model.h"

namespace mi::cli {

namespace fs = std::filesystem;

int cmd_index(const std::vector<std::string>& args) {
  fs::path bundle, out;
  weights::BuildOptions options;
  for (std::size_t i = 0; i < args.size(); ++i) {
    if (args[i] == "-o" && i + 1 < args.size()) out = path_arg(args[++i], "-o");
    else if (args[i] == "--checksum") options.checksum = true;
    else if (bundle.empty() && !args[i].empty() && args[i][0] != '-') bundle = path_arg(args[i], "bundle_dir");
    else {
      std::cerr << "usage: moss-infer index <bundle_dir> [-o <path>] [--checksum]\n";
      return 2;
    }
  }
  if (bundle.empty()) {
    std::cerr << "usage: moss-infer index <bundle_dir> [-o <path>] [--checksum]\n";
    return 2;
  }
  if (out.empty()) out = bundle / "moss.mossidx";
  if (!fs::is_directory(bundle)) throw std::invalid_argument("model bundle not found: " + util::path_utf8(bundle));
  weights::check_supported_bundle(bundle);
  weights::ModelIndex index = weights::build_index(bundle, options);
  weights::write_index(index, out);
  std::cout << "wrote " << util::path_utf8(out) << "\n";
  weights::print_summary(index, std::cout);
  return 0;
}

int cmd_inspect(const std::vector<std::string>& args) {
  if (args.size() != 1) {
    std::cerr << "usage: moss-infer inspect <bundle_dir|.mossidx>\n";
    return 2;
  }
  const fs::path target = path_arg(args[0], "target");
  weights::ModelIndex index = fs::is_directory(target) ? weights::build_index(target) : weights::load_index(target);
  weights::print_summary(index, std::cout);
  return 0;
}

}  // namespace mi::cli
