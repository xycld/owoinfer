// Host-side description of a varlen ("unpadded") batch and the optional debug validation of the
// cu_seqlens / max_seqlen contract shared by attention::flash_fwd and gdn::chunk_gated_delta_rule.
//
// CONTRACT (the kernels trust it; a violation silently corrupts or truncates the output):
//   * cu_seqlens is int32 [N+1] on the device, cu[0] = 0, non-decreasing, cu[N] == number of token rows T.
//   * max_seqlen (a HOST value, it sizes the launch grid) is >= every sequence length cu[b+1] - cu[b].
//     A too-small max_seqlen drops the tail of the longest sequences without any error.
// Varlen::from_lengths builds both consistently; use it, or keep the three values in sync yourself.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <span>
#include <vector>

#include "core/tensor.h"

namespace mi {

struct Varlen {
  std::vector<std::int32_t> cu_host;  // [N+1] prefix sums, cu_host[0] = 0
  int max_seqlen = 0;                 // longest sequence (0 if all are empty)
  int total = 0;                      // cu_host.back(): total token rows

  // Lengths may be 0; throws std::invalid_argument on a negative length or if the total exceeds INT32_MAX.
  static Varlen from_lengths(std::span<const int> lengths);

  int batch() const { return static_cast<int>(cu_host.size()) - 1; }
  std::size_t cu_bytes() const { return cu_host.size() * sizeof(std::int32_t); }
  // Copies cu_host to `device_cu` (>= cu_bytes() of device memory). The pageable host source is staged
  // before the call returns, so the Varlen need not outlive the copy.
  void upload_cu(void* device_cu, cudaStream_t stream) const;
};

// Debug validation switch. Default: on iff the library was compiled with -DMOSSINFER_DEBUG_CHECKS,
// otherwise off; toggle at run time with set_varlen_debug_checks(). Thread-safe.
bool varlen_debug_checks();
void set_varlen_debug_checks(bool on);

// Copies the device cu_seqlens to the host (synchronizing `stream`) and checks the contract above:
// cu[0] == 0, non-decreasing, cu[N] == expected_tokens, longest sequence <= max_seqlen. Throws
// std::invalid_argument naming `who`. Does nothing when the stream is being captured (a copy and sync
// would invalidate the capture), so validate once outside capture. Only call when varlen_debug_checks().
void validate_cu_seqlens(const Tensor& cu, int max_seqlen, std::int64_t expected_tokens, cudaStream_t stream,
                         const char* who);

}  // namespace mi
