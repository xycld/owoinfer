// Tensor exchange of a component denoise step: a host's device tensors handed over between its stream and the engine's
// by events, or host tensors through the session's pinned staging. No allocation per call beyond what is named here.
//
// Device mode, one call (DevicePointerCheck on every tensor first):
//   StreamHandoff::acquire   cudaEventRecord(ev_in, host stream); cudaStreamWaitEvent(engine stream, ev_in)
//   the forward (and the guided combine) on the engine stream, reading x and writing the outputs in place
//   StreamHandoff::release   cudaEventRecord(ev_out, engine stream); cudaStreamWaitEvent(host stream, ev_out);
//                            cudaEventSynchronize(ev_out): the call returns with its outputs written
// Host mode, one call: the tensors are the staging's views (StagingLayout over a StagingBlock); async H2D of x (and
// the control latent), the forward, D2H of the outputs, then one synchronization.
#pragma once

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>

namespace mi::runtime {

// Checks a host's device tensors: [ptr, ptr + bytes) must be device memory of `device` that the engine's kernels may
// use, i.e. not managed, in the device's primary context (the CUDA runtime's, which torch and CuPy allocate in) or in
// none (context-independent memory: VMM, as torch's expandable segments, and memory pools), and inside one allocation.
// One driver call per check (cuPointerGetAttributes); where libcuda lacks it, the runtime's memory type and device
// only. Construct it while the engine's runtime holds the device's primary context (any time after its first CUDA
// call there): it looks the context's handle up once.
class DevicePointerCheck {
 public:
  explicit DevicePointerCheck(int device);
  // Throws std::invalid_argument naming `what` (e.g. "x", "out[1]") and the reason.
  void check(const void* ptr, std::size_t bytes, std::string_view what) const;
  int device() const { return device_; }

 private:
  int device_ = 0;
  void* primary_ = nullptr;  // CUcontext; nullptr when the driver could not tell (then not compared)
};

// The two events of the device-mode handoff, created once per session (timing disabled, on the current device).
class StreamHandoff {
 public:
  StreamHandoff();
  ~StreamHandoff();
  StreamHandoff(const StreamHandoff&) = delete;
  StreamHandoff& operator=(const StreamHandoff&) = delete;

  // Work enqueued on `engine` after this starts only after everything enqueued on `host` so far. `host` may be any
  // handle: nullptr (the legacy default stream), cudaStreamLegacy, cudaStreamPerThread or a created stream.
  void acquire(cudaStream_t host, cudaStream_t engine);
  // Work enqueued on `host` after this starts only after everything enqueued on `engine` so far; returns once that
  // work has finished.
  void release(cudaStream_t engine, cudaStream_t host);

 private:
  cudaEvent_t in_ = nullptr;
  cudaEvent_t out_ = nullptr;
};

// The staging cannot hold what a run needs: refused by the host budget, or the pinned allocation failed (the two are
// told apart by the message; both are out of memory to a caller).
struct StagingError : std::runtime_error {
  StagingError(const std::string& what, std::size_t requested_bytes) : std::runtime_error(what), requested(requested_bytes) {}
  std::size_t requested;
};

// A page-locked host block a step stages through: the session's own allocation (owned) or memory a caller locked and
// lends for a run (borrowed: never freed or unlocked here).
struct StagingBlock {
  std::byte* base = nullptr;
  std::size_t bytes = 0;
  bool owned = false;
};

// Where a step's tensors sit in a staging block: x, then each output, then the control latent, each at a kArenaAlign
// (256-byte) boundary.
inline constexpr std::size_t kMaxStagingOutputs = 8;
struct StagingLayout {
  std::size_t x = 0, x_bytes = 0;
  std::array<std::size_t, kMaxStagingOutputs> out{};
  std::size_t out_bytes = 0;  // each output
  int outputs = 0;
  std::size_t control = 0, control_bytes = 0;  // control_bytes 0: no control slot
  std::size_t bytes = 0;                       // the block bytes the layout needs
  // x_bytes: the latents (L latents of a batch); out_bytes: one output; outputs in [1, kMaxStagingOutputs].
  // Throws std::invalid_argument otherwise.
  static StagingLayout plan(std::size_t x_bytes, int outputs, std::size_t out_bytes, std::size_t control_bytes);
};

// The session's pinned staging: grown to the largest run, kept between runs, freed at session close (release()).
class PinnedStaging {
 public:
  PinnedStaging() = default;
  ~PinnedStaging();
  PinnedStaging(const PinnedStaging&) = delete;
  PinnedStaging& operator=(const PinnedStaging&) = delete;

  // At least `bytes` of pinned memory afterwards: a no-op when it already holds that much, else the block is replaced
  // (contents not kept) by one of exactly `bytes`. `budget`: the most host bytes the staging may hold (the caller's
  // host budget left for it). Throws StagingError when `bytes` exceeds the budget (the block kept) or cudaHostAlloc
  // fails (the old block was freed first, so the staging holds never more than the budget: empty then).
  void ensure(std::size_t bytes, std::size_t budget);
  // Frees the block (session close).
  void release();
  // Pinned host bytes held (counted in the session's host memory).
  std::size_t bytes() const { return block_.bytes; }
  StagingBlock block() const { return block_; }

 private:
  StagingBlock block_;
};

}  // namespace mi::runtime
