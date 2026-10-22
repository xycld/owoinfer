#include "models/moss_dit/dit_graph.h"

#include <iterator>

namespace mi::moss_dit {
namespace {

struct Fnv {
  std::uint64_t h = 1469598103934665603ull;
  void mix(std::uint64_t v) {
    for (int i = 0; i < 8; ++i) h = (h ^ ((v >> (8 * i)) & 0xffu)) * 1099511628211ull;
  }
};

}  // namespace

std::uint64_t StepGraphKey::hash() const {
  Fnv f;
  f.mix(static_cast<std::uint64_t>(height));
  f.mix(static_cast<std::uint64_t>(width));
  f.mix(text_lengths.size());
  for (std::int64_t l : text_lengths) f.mix(static_cast<std::uint64_t>(l));
  f.mix(skips.size());
  for (const auto& s : skips) f.mix(s.to_ullong());  // kNumBlocks <= 64
  f.mix(static_cast<std::uint64_t>(block_begin));
  f.mix(static_cast<std::uint64_t>(block_end));
  f.mix(addresses.size());
  for (const void* p : addresses) f.mix(reinterpret_cast<std::uintptr_t>(p));
  f.mix(route_generation);
  f.mix(variant);
  return f.h;
}
static_assert(kNumBlocks <= 64, "StepGraphKey::hash packs a skip set into 64 bits");

bool StepGraphs::run(const StepGraphKey& key, cudaStream_t stream, const std::function<void(cudaStream_t)>& body) {
  const std::uint64_t h = key.hash();
  if (cache_.contains(h)) {
    if (!(keys_.at(h) == key)) {
      ++stats_.collisions;
      body(stream);
      return false;
    }
    ++stats_.replays;
    return cache_.launch(h, stream, body);
  }
  cache_.launch(h, stream, body);  // captures, may evict the least recently used entry
  ++stats_.captures;
  keys_[h] = key;
  for (auto it = keys_.begin(); it != keys_.end();) it = cache_.contains(it->first) ? std::next(it) : keys_.erase(it);
  return false;
}

void StepGraphs::clear() {
  cache_.clear();
  keys_.clear();
}

}  // namespace mi::moss_dit
