// `moss-infer quantize`: bf16 (or community-quantized) DiT -> INT8 ConvRot safetensors in ComfyUI's comfy_quant layout.
#include <algorithm>
#include <charconv>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include "tools/cli/commands.h"
#include "util/path.h"
#include "weights/quant/quant_convert.h"

namespace mi::cli {

namespace {

constexpr const char* kUsage =
    "usage: moss-infer quantize <transformer_dir|bundle_dir|dit.safetensors> -o <out.safetensors>\n"
    "         [--fuse-qkv] [--keep-bf16 <pattern,...>] [--lora <file> [--lora-scale <s>]]... [--group <G>]\n"
    "         [--comfy-attention-config] [--threads <n>]\n"
    "  --keep-bf16    comma-separated globs over GEMM names, e.g. 'layers.*.proj.ffn_down,layers.0.*'\n"
    "  --lora         kohya (diffusion_model.<m>.lora_{down,up}.weight + alpha) or PEFT (lora_A/lora_B) LoRA,\n"
    "                 merged in fp32 before quantizing; --lora-scale applies to the preceding --lora (default 1)\n"
    "  --group        ConvRot group of newly quantized GEMMs: 4, 16, 64, 256 (default) or 1024\n"
    "  --threads      CPU threads, 0 = all cores (default)\n";

std::vector<std::string> split_commas(const std::string& s) {
  std::vector<std::string> out;
  std::size_t b = 0;
  while (b <= s.size()) {
    const std::size_t e = std::min(s.find(',', b), s.size());
    if (e > b) out.push_back(s.substr(b, e - b));
    b = e + 1;
  }
  return out;
}

// The whole of `v` as a number, or nullopt.
template <typename T>
std::optional<T> parse_number(std::string_view v) {
  T x{};
  const auto r = std::from_chars(v.data(), v.data() + v.size(), x);
  if (v.empty() || r.ec != std::errc() || r.ptr != v.data() + v.size()) return std::nullopt;
  return x;
}

}  // namespace

int cmd_quantize(const std::vector<std::string>& args) {
  std::filesystem::path input, output;
  quant::ConvertOptions options;
  options.log = [](std::string_view s) { std::cout << s << std::endl; };
  auto bad_value = [](const std::string& flag, const std::string& v, const char* what) {
    std::cerr << "moss-infer quantize: " << flag << ": '" << v << "' is not " << what << "\n" << kUsage;
    return 2;
  };
  for (std::size_t i = 0; i < args.size(); ++i) {
    const std::string& a = args[i];
    const bool has_value = i + 1 < args.size();
    if (a == "-o" && has_value) {
      output = util::path_from_utf8(args[++i]);
    } else if (a == "--fuse-qkv") {
      options.fuse_qkv = true;
    } else if (a == "--comfy-attention-config") {
      options.attention_config = true;
    } else if (a == "--keep-bf16" && has_value) {
      for (auto& p : split_commas(args[++i])) options.keep_bf16.push_back(p);
    } else if (a == "--lora" && has_value) {
      options.loras.push_back({util::path_from_utf8(args[++i]), 1.0f});
    } else if (a == "--lora-scale" && has_value && !options.loras.empty()) {
      const auto v = parse_number<float>(args[++i]);
      if (!v || !std::isfinite(*v)) return bad_value(a, args[i], "a finite number");
      options.loras.back().strength = *v;
    } else if (a == "--group" && has_value) {
      const auto v = parse_number<int>(args[++i]);
      if (!v) return bad_value(a, args[i], "an integer");
      options.group = *v;
    } else if (a == "--threads" && has_value) {
      const auto v = parse_number<int>(args[++i]);
      if (!v || *v < 0) return bad_value(a, args[i], "a non-negative integer");
      options.threads = *v;
    } else if (input.empty() && !a.empty() && a[0] != '-') {
      input = util::path_from_utf8(a);
    } else {
      std::cerr << kUsage;
      return 2;
    }
  }
  if (input.empty() || output.empty()) {
    std::cerr << kUsage;
    return 2;
  }
  const quant::ConvertStats st = quant::convert(input, output, options);
  std::cout << "wrote " << util::path_utf8(output) << " (" << st.bytes_written << " bytes) in " << st.seconds << " s: " << st.quantized
            << " GEMMs quantized, " << st.copied << " copied, " << st.decoded << " decoded, " << st.kept_bf16 << " kept bf16";
  if (st.lora_modules > 0) std::cout << ", LoRA merged into " << st.lora_modules;
  std::cout << "\n";
  return 0;
}

}  // namespace mi::cli
