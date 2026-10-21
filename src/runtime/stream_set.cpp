#include "runtime/stream_set.h"

#include "core/cuda_check.h"

namespace mi::runtime {

EventPool::~EventPool() {
  for (cudaEvent_t e : free_) cudaEventDestroy(e);  // destructor: errors are not actionable here
}

cudaEvent_t EventPool::acquire() {
  std::lock_guard<std::mutex> lock(mu_);
  if (!free_.empty()) {
    cudaEvent_t e = free_.back();
    free_.pop_back();
    return e;
  }
  cudaEvent_t e = nullptr;
  MI_CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
  ++created_;
  return e;
}

void EventPool::release(cudaEvent_t e) {
  std::lock_guard<std::mutex> lock(mu_);
  free_.push_back(e);
}

std::size_t EventPool::created() const {
  std::lock_guard<std::mutex> lock(mu_);
  return created_;
}

StreamSet::StreamSet() {
  MI_CUDA_CHECK(cudaStreamCreateWithFlags(&compute_, cudaStreamNonBlocking));
  try {
    MI_CUDA_CHECK(cudaStreamCreateWithFlags(&copy_, cudaStreamNonBlocking));
  } catch (...) {
    cudaStreamDestroy(compute_);
    throw;
  }
}

StreamSet::~StreamSet() {
  cudaStreamDestroy(copy_);
  cudaStreamDestroy(compute_);
}

cudaEvent_t StreamSet::record(cudaStream_t s) {
  cudaEvent_t e = events_.acquire();
  try {
    MI_CUDA_CHECK(cudaEventRecord(e, s));
  } catch (...) {
    events_.release(e);
    throw;
  }
  return e;
}

void StreamSet::wait_event(cudaStream_t waiter, cudaEvent_t e) { MI_CUDA_CHECK(cudaStreamWaitEvent(waiter, e, 0)); }

void StreamSet::depend(cudaStream_t waiter, cudaStream_t producer) {
  cudaEvent_t e = record(producer);
  try {
    wait_event(waiter, e);
  } catch (...) {
    events_.release(e);
    throw;
  }
  events_.release(e);
}

}  // namespace mi::runtime
