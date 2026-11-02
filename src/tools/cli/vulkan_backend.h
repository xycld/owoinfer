// `moss-infer generate --backend vulkan`: text to image on the Vulkan backend (vulkan/), the fallback for GPUs the CUDA
// engine does not support. The engine's host code supplies what does not depend on the backend: the tokenizer and chat
// template, the seed's initial noise, the schedule and the PNG writer, so a seed gives the same inputs as on CUDA.
#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "pipeline/pipeline.h"

namespace mi::cli {

struct VulkanRun {
  int device = -1;      // -1: the first usable device, a discrete one first
  std::string kernels;  // "coop", "plain", or empty: coop where the device has bf16 cooperative matrices
  std::filesystem::path out_png;  // empty: no image written (the decode still runs)
  bool quiet = false;
};

// Whether this build has the Vulkan backend (MOSSINFER_VULKAN).
bool vulkan_backend_built();

// Runs txt2img for `req` (already validated) on Vulkan and writes the PNG. Throws std::invalid_argument for a request the
// backend does not support (it names the option), std::runtime_error for a device or model failure. Returns the final
// latents, f32 [128, height / 16, width / 16].
std::vector<float> run_vulkan_generate(const pipeline::Request& req, const VulkanRun& run);

}  // namespace mi::cli
