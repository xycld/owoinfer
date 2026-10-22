// Adapted from torchsde BrownianInterval (Apache-2.0, Copyright 2020 Google LLC); modified: ported to C++ for the
// BrownianTree configuration only (halfway tree, no Levy area, fixed tol 1e-6 / pool 24 / 45-entry cache), with fp32
// host arithmetic and torch-compatible CPU randn. The seed derivation reimplements NumPy's SeedSequence
// (BSD-3-Clause, Copyright (c) 2005-2026 NumPy Developers).
#include "pipeline/samplers/brownian_tree.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "pipeline/rng_torch_compat.h"

// fp32 tensor arithmetic of the reference, one rounding per op (cmake/modules/samplers.cmake: -ffp-contract=off).
#if defined(__FAST_MATH__) || defined(_M_FP_FAST)
#error "brownian_tree.cpp must not be built with fast-math"
#endif
#if defined(__clang__)
#pragma clang fp contract(off)
#endif

namespace mi::pipeline {
namespace {

// NumPy SeedSequence constants (numpy/random/bit_generator.pyx; BSD-3-Clause, Copyright (c) 2005-2026 NumPy
// Developers).
constexpr std::uint32_t kInitA = 0x43b0d7e5, kMultA = 0x931e8875, kInitB = 0x8b51f9dd, kMultB = 0x58f38ded;
constexpr std::uint32_t kMixMultL = 0xca01f9dd, kMixMultR = 0x4973f715;
constexpr int kXShift = 16;

std::uint32_t hashmix(std::uint32_t value, std::uint32_t& hash_const) {
  value ^= hash_const;
  hash_const *= kMultA;
  value *= hash_const;
  value ^= value >> kXShift;
  return value;
}

std::uint32_t mix(std::uint32_t x, std::uint32_t y) {
  std::uint32_t result = kMixMultL * x - kMixMultR * y;
  result ^= result >> kXShift;
  return result;
}

// _int_to_uint32_array: little-endian 32-bit words, [0] for 0.
void append_words(std::vector<std::uint32_t>& out, std::uint64_t n) {
  if (n == 0) {
    out.push_back(0);
    return;
  }
  while (n > 0) {
    out.push_back(static_cast<std::uint32_t>(n & 0xffffffffu));
    n >>= 32;
  }
}

// torchsde's tolerance and pool size for BrownianTree, and its LRU cache size.
constexpr double kTol = 1e-6;
constexpr int kRoundDigits = 6;  // -int(math.log10(tol))
constexpr int kPoolSize = 24;
constexpr std::size_t kCacheSize = 45;
static_assert(kTol == 1e-6);

}  // namespace

std::vector<std::uint32_t> seed_sequence_state(std::uint64_t entropy, std::span<const std::uint64_t> spawn_key,
                                               int pool_size, int n_words) {
  if (pool_size < 1 || n_words < 0) throw std::invalid_argument("seed_sequence_state: bad pool size or word count");
  // get_assembled_entropy: run entropy, zero-padded to the pool size when a spawn key follows.
  std::vector<std::uint32_t> ent;
  append_words(ent, entropy);
  if (!spawn_key.empty() && ent.size() < static_cast<std::size_t>(pool_size)) ent.resize(static_cast<std::size_t>(pool_size), 0);
  for (const std::uint64_t k : spawn_key) append_words(ent, k);

  // mix_entropy
  std::vector<std::uint32_t> pool(static_cast<std::size_t>(pool_size));
  std::uint32_t hash_const = kInitA;
  for (std::size_t i = 0; i < pool.size(); ++i) pool[i] = hashmix(i < ent.size() ? ent[i] : 0u, hash_const);
  for (std::size_t src = 0; src < pool.size(); ++src) {
    for (std::size_t dst = 0; dst < pool.size(); ++dst) {
      if (src != dst) pool[dst] = mix(pool[dst], hashmix(pool[src], hash_const));
    }
  }
  for (std::size_t src = pool.size(); src < ent.size(); ++src) {
    for (std::size_t dst = 0; dst < pool.size(); ++dst) pool[dst] = mix(pool[dst], hashmix(ent[src], hash_const));
  }

  // generate_state (uint32)
  std::vector<std::uint32_t> out(static_cast<std::size_t>(n_words));
  std::uint32_t hc = kInitB;
  for (std::size_t i = 0; i < out.size(); ++i) {
    std::uint32_t v = pool[i % pool.size()];
    v ^= hc;
    hc *= kMultB;
    v *= hc;
    v ^= v >> kXShift;
    out[i] = v;
  }
  return out;
}

double python_round(double x, int ndigits) {
  // CPython rounds through a correctly rounded decimal string (dtoa mode 3); glibc's printf is exact as well.
  if (!std::isfinite(x)) return x;
  char buf[512];
  const int n = std::snprintf(buf, sizeof buf, "%.*f", ndigits, x);
  if (n <= 0 || n >= static_cast<int>(sizeof buf)) throw std::runtime_error("python_round: value out of range");
  return std::strtod(buf, nullptr);
}

struct BrownianTreeNoise::Impl {
  struct Node {
    double start = 0, end = 0;
    Node* parent = nullptr;
    bool is_left = false;
    bool split = false;
    double midway = 0;
    std::uint64_t spawn_key = 0;
    int depth = 0;
    std::uint32_t w_seed = 0;
    std::unique_ptr<Node> left, right;
  };
  using Vec = std::shared_ptr<const std::vector<float>>;

  std::int64_t numel = 0;
  std::uint64_t entropy = 0;
  Node top;
  Vec top_w;
  Node* last = &top;
  std::unordered_map<const Node*, Vec> cache;
  std::deque<const Node*> order;

  static double round6(double x) { return python_round(x, kRoundDigits); }

  void split_exact(Node* node, double midway) {
    node->midway = round6(midway);
    if (node->parent != nullptr) {
      if (node->parent->spawn_key > (std::uint64_t{1} << 62)) throw std::runtime_error("BrownianTreeNoise: tree too deep");
      node->spawn_key = 2 * node->parent->spawn_key + (node->is_left ? 0 : 1);
      node->depth = node->parent->depth + 1;
    }
    const std::uint64_t key[2] = {node->spawn_key, static_cast<std::uint64_t>(node->depth)};
    node->w_seed = seed_sequence_state(entropy, key, kPoolSize, 4)[0];  // W, H, left A, right A seeds
    node->split = true;
    node->left = std::make_unique<Node>();
    node->left->start = node->start;
    node->left->end = round6(midway);
    node->left->parent = node;
    node->left->is_left = true;
    node->right = std::make_unique<Node>();
    node->right->start = round6(midway);
    node->right->end = node->end;
    node->right->parent = node;
  }

  // _split with halfway_tree: dyadic halves until `midway` is a split point.
  void split(Node* node, double midway) {
    for (;;) {
      split_exact(node, 0.5 * (node->end + node->start));
      if (midway > node->midway) node = node->right.get();
      else if (midway < node->midway) node = node->left.get();
      else return;
    }
  }

  void loc_inner(Node* node, double ta, double tb, std::vector<Node*>& out) {
    for (;;) {
      if (ta < node->start || tb > node->end) {
        node = node->parent;
        continue;
      }
      if (ta == node->start && tb == node->end) {
        out.push_back(node);
        return;
      }
      if (!node->split) {
        if (ta == node->start) {
          split(node, tb);
          node = node->left.get();
        } else {
          split(node, ta);
          node = node->right.get();
        }
        continue;
      }
      if (tb <= node->midway) {
        node = node->left.get();
        continue;
      }
      if (ta >= node->midway) {
        node = node->right.get();
        continue;
      }
      loc_inner(node->left.get(), ta, node->midway, out);
      ta = node->midway;
      node = node->right.get();
    }
  }

  void remember(const Node* node, Vec v) {
    if (cache.count(node) != 0) {
      for (auto it = order.begin(); it != order.end(); ++it) {
        if (*it == node) {
          order.erase(it);
          break;
        }
      }
    } else if (cache.size() >= kCacheSize) {
      cache.erase(order.front());
      order.pop_front();
    }
    cache[node] = v;
    order.push_back(node);
  }

  // W over the node's interval: the Brownian bridge from the parent's increment.
  Vec increment(const Node* node) {
    if (node == &top) return top_w;
    if (auto it = cache.find(node); it != cache.end()) return it->second;
    const Node* p = node->parent;
    const Vec w = increment(p);
    const double h_reciprocal = 1 / (p->end - p->start);
    const double left_diff = p->midway - p->start;
    const double right_diff = p->end - p->midway;
    const double var = left_diff * right_diff * h_reciprocal;
    const float c_left = static_cast<float>(left_diff), c_h = static_cast<float>(h_reciprocal);
    const float c_std = static_cast<float>(std::sqrt(var));
    const std::vector<float> noise = rng::torch_cpu_randn(p->w_seed, numel);
    auto out = std::make_shared<std::vector<float>>(static_cast<std::size_t>(numel));
    for (std::size_t i = 0; i < out->size(); ++i) {
      const float mean = ((*w)[i] * c_left) * c_h;
      const float left_w = mean + c_std * noise[i];
      (*out)[i] = node->is_left ? left_w : (*w)[i] - left_w;
    }
    Vec v = out;
    remember(node, v);
    return v;
  }

  // BrownianInterval.__call__(ta, tb) for ta <= tb.
  std::vector<float> query(double ta, double tb) {
    ta = std::max(ta, top.start);
    tb = std::max(tb, top.start);
    ta = std::min(ta, top.end);
    tb = std::min(tb, top.end);
    if (ta > tb) throw std::invalid_argument("BrownianTreeNoise: query times out of order");
    if (ta == tb) return std::vector<float>(static_cast<std::size_t>(numel), 0.0f);
    std::vector<Node*> intervals;
    loc_inner(last, round6(ta), round6(tb), intervals);
    last = intervals.back();
    std::vector<float> w = *increment(intervals.front());
    for (std::size_t k = 1; k < intervals.size(); ++k) {
      const Vec wi = increment(intervals[k]);
      for (std::size_t i = 0; i < w.size(); ++i) w[i] = w[i] + (*wi)[i];
    }
    return w;
  }
};

BrownianTreeNoise::BrownianTreeNoise(std::int64_t numel, float sigma_min, float sigma_max, std::uint64_t seed)
    : impl_(std::make_unique<Impl>()) {
  if (numel < 1 || !(sigma_min < sigma_max) || !std::isfinite(sigma_min) || !std::isfinite(sigma_max)) {
    throw std::invalid_argument("BrownianTreeNoise: need numel >= 1 and finite sigma_min < sigma_max");
  }
  Impl& m = *impl_;
  m.numel = numel;
  m.entropy = seed;
  const double t0 = sigma_min, t1 = sigma_max;
  m.top.start = Impl::round6(t0);
  m.top.end = Impl::round6(t1);
  // initial W seed (then H and the top Levy-area seed, unused without Levy area).
  const std::uint32_t w_seed = seed_sequence_state(seed, {}, kPoolSize, 3)[0];
  std::vector<float> w = rng::torch_cpu_randn(w_seed, numel);
  const float scale = static_cast<float>(std::sqrt(t1 - t0));  // unrounded endpoints, as the reference
  for (float& v : w) v = v * scale;
  m.top_w = std::make_shared<const std::vector<float>>(std::move(w));
}

BrownianTreeNoise::~BrownianTreeNoise() = default;
BrownianTreeNoise::BrownianTreeNoise(BrownianTreeNoise&&) noexcept = default;
BrownianTreeNoise& BrownianTreeNoise::operator=(BrownianTreeNoise&&) noexcept = default;

std::vector<float> BrownianTreeNoise::noise(float sigma, float sigma_next) {
  // BatchedBrownianTree: query the sorted interval, flip the sign for a decreasing one, then divide by
  // sqrt(|sigma_next - sigma|) computed in fp32.
  const bool decreasing = sigma > sigma_next;
  const float lo = decreasing ? sigma_next : sigma, hi = decreasing ? sigma : sigma_next;
  std::vector<float> w = impl_->query(lo, hi);
  const float d = std::sqrt(std::fabs(sigma_next - sigma));
  for (float& v : w) v = (decreasing ? -v : v) / d;
  return w;
}

}  // namespace mi::pipeline
