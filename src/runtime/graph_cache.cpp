#include "runtime/graph_cache.h"

#include <stdexcept>

#include "core/cuda_check.h"

namespace mi::runtime {

GraphCache::GraphCache(std::size_t capacity) : capacity_(capacity) {
  if (capacity == 0) throw std::invalid_argument("GraphCache capacity must be positive");
}

GraphCache::~GraphCache() { clear(); }

void GraphCache::clear() {
  // Destroying an exec that was already launched is safe; the runtime defers teardown until it finishes.
  for (Entry& e : lru_) cudaGraphExecDestroy(e.exec);  // errors are not actionable here
  lru_.clear();
  map_.clear();
}

bool GraphCache::launch(std::uint64_t key, cudaStream_t stream, const std::function<void(cudaStream_t)>& body) {
  if (auto it = map_.find(key); it != map_.end()) {
    lru_.splice(lru_.begin(), lru_, it->second);
    ++stats_.hits;
    MI_CUDA_CHECK(cudaGraphLaunch(it->second->exec, stream));
    return true;
  }
  ++stats_.misses;
  MI_CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  try {
    body(stream);
  } catch (...) {
    cudaGraph_t discard = nullptr;
    if (cudaStreamEndCapture(stream, &discard) == cudaSuccess && discard != nullptr) cudaGraphDestroy(discard);
    cudaGetLastError();  // capture invalidation errors are expected on this path
    throw;
  }
  cudaGraph_t graph = nullptr;
  MI_CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
  cudaGraphExec_t exec = nullptr;
  cudaError_t err = cudaGraphInstantiate(&exec, graph, 0);
  cudaGraphDestroy(graph);
  MI_CUDA_CHECK(err);

  if (lru_.size() >= capacity_) {
    cudaGraphExecDestroy(lru_.back().exec);
    map_.erase(lru_.back().key);
    lru_.pop_back();
    ++stats_.evictions;
  }
  lru_.push_front({key, exec});
  map_[key] = lru_.begin();
  MI_CUDA_CHECK(cudaGraphLaunch(exec, stream));
  return false;
}

}  // namespace mi::runtime
