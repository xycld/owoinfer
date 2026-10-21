// CUDA graphs of DiT step work for resident weights (P5 step 31).
//
// A graph bakes in kernel arguments and device addresses, so its key is the exact step: latent grid, every row's
// text length and skip set (text lengths change the joint attention and the token offsets: they cannot be bucketed or
// padded), the block range the graph covers (a step split into segments keeps cancel latency at one segment), and
// every device address the captured work binds that can change between calls (inputs, outputs, the arena stack base).
// Entries live in an LRU (runtime::GraphCache). The cache is indexed by a 64-bit hash and the full key is compared on
// every hit, so a hash collision runs the work eagerly instead of replaying the wrong graph.
//
// Streamed blocks (P4) wait on host-driven copies and cannot be captured: graphs are for resident blocks only.
// Not thread-safe (one per model).
#pragma once

#include <cuda_runtime.h>

#include <bitset>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <vector>

#include "models/moss_dit/dit_weights.h"
#include "runtime/graph_cache.h"

namespace mi::moss_dit {

struct StepGraphKey {
  std::int64_t height = 0, width = 0;               // latent grid
  std::vector<std::int64_t> text_lengths;           // per row, in row order
  std::vector<std::bitset<kNumBlocks>> skips;       // per row
  int block_begin = 0, block_end = kNumBlocks;      // blocks [begin, end) captured
  std::vector<const void*> addresses;               // device addresses the work binds

  bool operator==(const StepGraphKey&) const = default;
  std::uint64_t hash() const;
};

struct StepGraphStats {
  std::uint64_t replays = 0;
  std::uint64_t captures = 0;
  std::uint64_t collisions = 0;  // ran eagerly: another key with the same hash is cached
};

class StepGraphs {
 public:
  explicit StepGraphs(std::size_t capacity) : cache_(capacity) {}

  // Enqueues body(stream)'s work on `stream` (non-default) through the graph of `key`: replayed when cached, otherwise
  // captured from one call of body and then launched. body must only enqueue work on `stream` (no host
  // synchronization, no allocation) and must enqueue the same work for the same key. Returns true on a replay.
  bool run(const StepGraphKey& key, cudaStream_t stream, const std::function<void(cudaStream_t)>& body);

  const StepGraphStats& stats() const { return stats_; }
  std::size_t size() const { return cache_.size(); }
  void clear();

 private:
  runtime::GraphCache cache_;
  std::unordered_map<std::uint64_t, StepGraphKey> keys_;  // the full key of every cached hash
  StepGraphStats stats_;
};

}  // namespace mi::moss_dit
