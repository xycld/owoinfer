// Brownian-tree noise for SDE samplers, bit-exact with the k-diffusion family's BrownianTreeNoiseSampler on the CPU
// (torchsde 0.2.6 BrownianTree: a BrownianInterval with halfway_tree, tol 1e-6, pool_size 24), so DPM++ SDE samplers
// reproduce that reference's noise for a given seed.
//
// The process W on [t0, t1] = [sigma_min, sigma_max] is built from a dyadic tree of intervals (endpoints rounded to
// 6 decimals, as Python's round(x, 6)). Each split interval gets four 32-bit seeds from
// numpy.random.SeedSequence(entropy=seed, spawn_key=(key, depth), pool_size=24).generate_state(4), where key is the
// interval's path in the tree (root 0, child = 2 * parent + right) and depth its level; its children's increments
// come from the Brownian bridge
//   W_left = (l / h) * W + sqrt(l * r / h) * randn(W_seed),  W_right = W - W_left     (l, r: child widths, h = l + r)
// with randn = torch's CPU generator (pipeline/rng_torch_compat.h) and the arithmetic in fp32 as torch's CPU tensor
// ops do it (Python floats cast to fp32 at each op). A query (ta, tb) is resolved by the reference's search from the
// last queried interval, splitting intervals as needed, and sums the increments of the intervals it covers, so the
// noise depends on the sequence of queries, as in the reference. The noise of a step is
//   -W(sigma_next, sigma) / sqrt(|sigma_next - sigma|)       (sigma > sigma_next).
// Increments of recently used intervals are cached (45 entries, as torchsde), which only affects speed. Host memory:
// up to 46 latent-sized fp32 vectors (the cache plus the root increment), ~94 MB at 1024x1024 and ~380 MB at
// 2048x2048; a query splits up to ~20 intervals, each one torch CPU randn of the latent size.
#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace mi::pipeline {

// numpy.random.SeedSequence(entropy, spawn_key=spawn_key, pool_size=pool_size).generate_state(n_words) for a
// non-negative integer entropy and spawn key.
std::vector<std::uint32_t> seed_sequence_state(std::uint64_t entropy, std::span<const std::uint64_t> spawn_key,
                                               int pool_size, int n_words);

// Python's round(x, ndigits) for a double (correctly rounded to `ndigits` decimals).
double python_round(double x, int ndigits);

class BrownianTreeNoise {
 public:
  // BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed, cpu=True) for a latent of `numel` elements;
  // sigma_min < sigma_max. Throws std::invalid_argument.
  BrownianTreeNoise(std::int64_t numel, float sigma_min, float sigma_max, std::uint64_t seed);
  ~BrownianTreeNoise();
  BrownianTreeNoise(BrownianTreeNoise&&) noexcept;
  BrownianTreeNoise& operator=(BrownianTreeNoise&&) noexcept;

  // The sampler's noise for one step from sigma to sigma_next (numel values, fp32, unit variance).
  std::vector<float> noise(float sigma, float sigma_next);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace mi::pipeline
