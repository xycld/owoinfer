// `moss-infer quantize`: bf16, community-quantized or GGUF DiT -> INT8 ConvRot safetensors in ComfyUI's comfy_quant layout,
// or (--format nvfp4) bf16 DiT -> mixed NVFP4 + INT8 ConvRot (weights/quant/quant_convert.h).
#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "pipeline/nvfp4_calibration.h"
#include "tools/cli/commands.h"
#include "util/path.h"
#include "util/regular_file.h"
#include "weights/quant/quant_convert.h"

namespace mi::cli {

namespace {

constexpr const char* kUsage =
    "usage: moss-infer quantize <transformer_dir|bundle_dir|dit.safetensors|dit.gguf> -o <out.safetensors>\n"
    "         [--fuse-qkv] [--keep-bf16 <pattern,...>] [--lora <file> [--lora-scale <s>]]... [--group <G>]\n"
    "         [--comfy-attention-config] [--threads <n>]\n"
    "         [--format int8|nvfp4] [--recipe rtn|mse|gptq|gptq-mse] [--keep <preset|units>] [--calib-model <bundle_dir>]\n"
    "  --keep-bf16    comma-separated globs over GEMM names, e.g. 'layers.*.proj.ffn_down,layers.0.*'\n"
    "  --lora         kohya (diffusion_model.<m>.lora_{down,up}.weight + alpha) or PEFT (lora_A/lora_B) LoRA,\n"
    "                 merged in fp32 before quantizing; --lora-scale applies to the preceding --lora (default 1)\n"
    "  --group        ConvRot group of newly quantized GEMMs: 4, 16, 64, 256 (default) or 1024\n"
    "  --threads      CPU threads, 0 = all cores (default)\n"
    "  --format       int8 (default): INT8 ConvRot; nvfp4: NVFP4 block GEMMs (RTX 50), the --keep units INT8 ConvRot,\n"
    "                 q/k/v fused, from bf16 weights\n"
    "  --recipe       nvfp4 weights: gptq-mse (default), gptq, mse or rtn (bit-exact to comfy-kitchen); gptq*\n"
    "                 capture calibration Hessians with the engine on the GPU (3 built-in prompts)\n"
    "  --keep         units kept INT8: g0.9 (default, 42 units), g0.95 (21), none, or e.g. 'b37.down,b26.o'\n"
    "  --calib-model  the bundle (tokenizer, text encoder) gptq* calibrate with; default: the input's bundle when the\n"
    "                 input is a bundle or its transformer directory (required for a single-file input)\n"
    "  --hessians-out <file>  gptq*: capture the calibration Hessians (GPU) into <file> and stop (no -o)\n"
    "  --hessians <file>      gptq*: take the Hessians from such a file instead of capturing (CPU only)\n";

// The bundle a diffusers input belongs to: the directory itself when it has a transformer/ subdirectory, else its
// parent when the input is that transformer directory. Empty for a single file or anything else.
std::filesystem::path bundle_of(const std::filesystem::path& input) {
  if (!util::file_info(input).directory()) return {};
  if (util::file_info(input / "transformer").directory()) return input;
  std::error_code ec;
  // "x/transformer/" and "." normalize to a path without a file name: drop the separator before taking the parent.
  std::filesystem::path dir = std::filesystem::absolute(input, ec).lexically_normal();
  if (!dir.has_filename()) dir = dir.parent_path();
  const std::filesystem::path parent = dir.parent_path();
  if (const util::FileInfo t = util::file_info(parent / "transformer"); t.directory() && util::same_file(t, util::file_info(input))) {
    return parent;
  }
  return {};
}

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
  std::string format = "int8", recipe_arg, keep_arg;
  std::filesystem::path calib_model, hessians_out, hessians_in;
  for (std::size_t i = 0; i < args.size(); ++i) {
    const std::string& a = args[i];
    const bool has_value = i + 1 < args.size();
    if (a == "--format" && has_value) {
      format = args[++i];
      if (format != "int8" && format != "nvfp4") return bad_value(a, format, "int8 or nvfp4");
    } else if (a == "--recipe" && has_value) {
      recipe_arg = args[++i];
      if (!quant::parse_nvfp4_recipe(recipe_arg)) return bad_value(a, recipe_arg, "rtn, mse, gptq or gptq-mse");
    } else if (a == "--keep" && has_value) {
      keep_arg = args[++i];
      if (keep_arg.empty()) return bad_value(a, keep_arg, "a preset (none, g0.9, g0.95) or a list of units");
    } else if (a == "--calib-model" && has_value) {
      calib_model = path_arg(args[++i], "--calib-model");
    } else if (a == "--hessians-out" && has_value) {
      hessians_out = path_arg(args[++i], "--hessians-out");
    } else if (a == "--hessians" && has_value) {
      hessians_in = path_arg(args[++i], "--hessians");
    } else if (a == "-o" && has_value) {
      output = path_arg(args[++i], "-o");
    } else if (a == "--fuse-qkv") {
      options.fuse_qkv = true;
    } else if (a == "--comfy-attention-config") {
      options.attention_config = true;
    } else if (a == "--keep-bf16" && has_value) {
      for (auto& p : split_commas(args[++i])) options.keep_bf16.push_back(p);
    } else if (a == "--lora" && has_value) {
      options.loras.push_back({path_arg(args[++i], "--lora"), 1.0f});
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
      input = path_arg(a, "input");
    } else {
      std::cerr << kUsage;
      return 2;
    }
  }
  if (input.empty() || output.empty() == hessians_out.empty()) {
    std::cerr << kUsage;
    return 2;
  }
  if (format != "nvfp4" && (!recipe_arg.empty() || !keep_arg.empty() || !calib_model.empty() || !hessians_out.empty() || !hessians_in.empty())) {
    std::cerr << "moss-infer quantize: --recipe, --keep, --calib-model and --hessians[-out] need --format nvfp4\n" << kUsage;
    return 2;
  }
  if (format == "nvfp4") {
    quant::Nvfp4Options fp4;
    if (!recipe_arg.empty()) fp4.recipe = *quant::parse_nvfp4_recipe(recipe_arg);
    fp4.keep = quant::parse_nvfp4_keep(keep_arg.empty() ? "g0.9" : keep_arg, &fp4.keep_name);
    const bool gptq = quant::needs_hessian(fp4.recipe);
    if ((!hessians_out.empty() || !hessians_in.empty()) && (!gptq || (!hessians_out.empty() && !hessians_in.empty()))) {
      std::cerr << "moss-infer quantize: one of --hessians-out / --hessians, with a gptq recipe\n";
      return 2;
    }
    if (!hessians_in.empty() && !calib_model.empty()) {
      std::cerr << "moss-infer quantize: --calib-model has no effect with --hessians (the file holds the capture)\n";
      return 2;
    }
    if (gptq && !options.loras.empty()) {
      std::cerr << "moss-infer quantize: --lora cannot be used with recipe " << quant::recipe_name(fp4.recipe)
                << " (its calibration capture runs the model without the LoRA); use --recipe rtn or mse, or convert a merged bf16 file\n";
      return 2;
    }
    if (!hessians_in.empty()) {
      fp4.hessian_file = hessians_in;
    } else if (gptq) {
      pipeline::CalibrationOptions co;
      co.model = calib_model.empty() ? bundle_of(input) : calib_model;
      if (co.model.empty()) {
        std::cerr << "moss-infer quantize: recipe " << quant::recipe_name(fp4.recipe)
                  << " (the default) calibrates with the bundle's text encoder, which a single-file input does not have: pass "
                     "--calib-model <bundle_dir>, or choose --recipe mse (no calibration)\n";
        return 2;
      }
      // A single-file DiT is captured as it is; a diffusers input only with its own bundle's transformer.
      if (!util::file_info(input).directory()) {
        co.dit_file = input;
      } else if (bundle_of(input).empty() ||
                 !util::same_file(util::file_info(bundle_of(input)), util::file_info(co.model))) {
        std::cerr << "moss-infer quantize: for a diffusers input, --calib-model must be the input's own bundle\n";
        return 2;
      }
      co.log = options.log;
      fp4.hessians = pipeline::nvfp4_hessian_source(std::move(co));
    }
    options.nvfp4 = std::move(fp4);
  }
  if (!hessians_out.empty()) {
    const auto t0 = std::chrono::steady_clock::now();
    const quant::HessianFileInfo info = quant::write_hessian_file(input, hessians_out, options);
    std::cout << "wrote " << util::path_utf8(hessians_out) << ": " << info.units << " Hessians in "
              << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() << " s; capture " << info.capture.capture_class << "\n";
    return 0;
  }
  const quant::ConvertStats st = quant::convert(input, output, options);
  std::cout << "wrote " << util::path_utf8(output) << " (" << st.bytes_written << " bytes) in " << st.seconds << " s: " << st.quantized
            << " GEMMs quantized";
  if (options.nvfp4) std::cout << " (" << st.nvfp4 << " nvfp4, recipe " << quant::recipe_name(options.nvfp4->recipe) << ")";
  std::cout << ", " << st.copied << " copied, " << st.decoded << " decoded, " << st.kept_bf16 << " kept bf16";
  if (st.lora_modules > 0) std::cout << ", LoRA merged into " << st.lora_modules;
  std::cout << "\n";
  return 0;
}

}  // namespace mi::cli
