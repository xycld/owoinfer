#include "pipeline/samplers/initial_noise.h"

#include <stdexcept>
#include <string>

#include "pipeline/rng_torch_compat.h"

namespace mi::pipeline {

const char* noise_layout_name(NoiseLayout layout) {
  return layout == NoiseLayout::kBatchStream ? "batch_stream" : "official";
}

NoiseLayout parse_noise_layout(std::string_view name) {
  if (name == "official") return NoiseLayout::kOfficial;
  if (name == "batch_stream") return NoiseLayout::kBatchStream;
  throw std::invalid_argument("unknown noise layout '" + std::string(name) + "' (expected official or batch_stream)");
}

std::vector<float> initial_noise(NoiseLayout layout, std::uint64_t seed, std::int64_t channels, std::int64_t latent_h,
                                 std::int64_t latent_w, int batch_index, int batch_size) {
  if (channels < 1 || latent_h < 1 || latent_w < 1 || batch_size < 1 || batch_index < 0 || batch_index >= batch_size) {
    throw std::invalid_argument("initial_noise: sizes must be positive and batch_index in [0, batch_size)");
  }
  const std::int64_t n = channels * latent_h * latent_w;
  if (layout == NoiseLayout::kOfficial || batch_size == 1) return rng::torch_cpu_randn(seed, n);
  // One draw for the whole batch, as torch.randn([B, C, h, w]).
  const std::vector<float> all = rng::torch_cpu_randn(seed, n * batch_size);
  const auto begin = all.begin() + static_cast<std::ptrdiff_t>(n * batch_index);
  return {begin, begin + static_cast<std::ptrdiff_t>(n)};
}

}  // namespace mi::pipeline
