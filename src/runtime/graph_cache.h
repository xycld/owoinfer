// CUDA graph capture/replay cache keyed by a caller-chosen 64-bit key (e.g. a hash of shape bucket
// and step kind). The first launch() for a key captures the callback's stream work and instantiates
// it; later launches replay the executable graph. Least-recently-used entries are destroyed when the
// capacity is exceeded.
//
// A captured graph bakes in kernel arguments and device pointers, so the key must determine them
// (same arena addresses, same shapes). Not thread-safe.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <list>
#include <unordered_map>

namespace mi::runtime {

struct GraphCacheStats {
  std::uint64_t hits = 0;
  std::uint64_t misses = 0;
  std::uint64_t evictions = 0;
};

class GraphCache {
 public:
  explicit GraphCache(std::size_t capacity);
  ~GraphCache();
  GraphCache(const GraphCache&) = delete;
  GraphCache& operator=(const GraphCache&) = delete;

  // Replays the graph for `key` on `stream`, capturing `body(stream)` first if absent. `stream` must
  // be a non-default stream. Returns true on a cache hit. If `body` throws or capture fails, nothing is
  // cached and the capture is ended before the exception propagates.
  bool launch(std::uint64_t key, cudaStream_t stream, const std::function<void(cudaStream_t)>& body);

  bool contains(std::uint64_t key) const { return map_.count(key) != 0; }
  std::size_t size() const { return map_.size(); }
  std::size_t capacity() const { return capacity_; }
  const GraphCacheStats& stats() const { return stats_; }
  void clear();

 private:
  struct Entry {
    std::uint64_t key;
    cudaGraphExec_t exec;
  };
  std::size_t capacity_;
  std::list<Entry> lru_;  // front = most recently used
  std::unordered_map<std::uint64_t, std::list<Entry>::iterator> map_;
  GraphCacheStats stats_;
};

}  // namespace mi::runtime
