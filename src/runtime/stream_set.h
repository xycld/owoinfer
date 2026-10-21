// The streams and events one engine instance schedules on: a compute stream, a copy stream for
// H2D weight traffic, and a pool of timing-disabled events for cross-stream dependencies.
#pragma once

#include <cuda_runtime.h>

#include <mutex>
#include <vector>

namespace mi::runtime {

// Events are cheap but not free to create; reuse them. Thread-safe.
class EventPool {
 public:
  EventPool() = default;
  ~EventPool();
  EventPool(const EventPool&) = delete;
  EventPool& operator=(const EventPool&) = delete;

  cudaEvent_t acquire();
  // Safe right after cudaStreamWaitEvent / a query: waits bind to the record that was current when
  // they were issued, so a later re-record of the event does not affect them.
  void release(cudaEvent_t e);
  std::size_t created() const;

 private:
  mutable std::mutex mu_;
  std::vector<cudaEvent_t> free_;
  std::size_t created_ = 0;
};

class StreamSet {
 public:
  StreamSet();
  ~StreamSet();
  StreamSet(const StreamSet&) = delete;
  StreamSet& operator=(const StreamSet&) = delete;

  cudaStream_t compute() const { return compute_; }
  cudaStream_t copy() const { return copy_; }
  EventPool& events() { return events_; }

  // Records a pooled event on `s`; the caller releases it to events() once no longer needed.
  cudaEvent_t record(cudaStream_t s);
  // Work enqueued on `waiter` after this call starts only after everything already enqueued on `producer`.
  void depend(cudaStream_t waiter, cudaStream_t producer);
  // Same, but against an event the caller recorded earlier (e.g. a weight group's copy-complete event).
  static void wait_event(cudaStream_t waiter, cudaEvent_t e);

 private:
  cudaStream_t compute_ = nullptr;
  cudaStream_t copy_ = nullptr;
  EventPool events_;
};

}  // namespace mi::runtime
