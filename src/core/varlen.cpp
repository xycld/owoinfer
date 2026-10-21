#include "core/varlen.h"

#include <algorithm>
#include <atomic>
#include <limits>
#include <stdexcept>
#include <string>

#include "core/cuda_check.h"

namespace mi {

namespace {
#ifdef MOSSINFER_DEBUG_CHECKS
constexpr bool kDefaultChecks = true;
#else
constexpr bool kDefaultChecks = false;
#endif
std::atomic<bool> g_checks{kDefaultChecks};
}  // namespace

Varlen Varlen::from_lengths(std::span<const int> lengths) {
  Varlen v;
  v.cu_host.reserve(lengths.size() + 1);
  v.cu_host.push_back(0);
  std::int64_t sum = 0;
  for (int n : lengths) {
    if (n < 0) throw std::invalid_argument("Varlen: negative sequence length " + std::to_string(n));
    sum += n;
    if (sum > std::numeric_limits<std::int32_t>::max()) throw std::invalid_argument("Varlen: more than INT32_MAX tokens");
    v.cu_host.push_back(static_cast<std::int32_t>(sum));
    v.max_seqlen = std::max(v.max_seqlen, n);
  }
  v.total = static_cast<int>(sum);
  return v;
}

void Varlen::upload_cu(void* device_cu, cudaStream_t stream) const {
  MI_CUDA_CHECK(cudaMemcpyAsync(device_cu, cu_host.data(), cu_bytes(), cudaMemcpyHostToDevice, stream));
}

bool varlen_debug_checks() { return g_checks.load(std::memory_order_relaxed); }
void set_varlen_debug_checks(bool on) { g_checks.store(on, std::memory_order_relaxed); }

void validate_cu_seqlens(const Tensor& cu, int max_seqlen, std::int64_t expected_tokens, cudaStream_t stream,
                         const char* who) {
  cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
  MI_CUDA_CHECK(cudaStreamIsCapturing(stream, &status));
  if (status != cudaStreamCaptureStatusNone) return;
  const auto fail = [&](const std::string& what) {
    throw std::invalid_argument(std::string(who) + ": cu_seqlens contract violated: " + what);
  };
  std::vector<std::int32_t> h(static_cast<std::size_t>(cu.dim(0)));
  MI_CUDA_CHECK(cudaMemcpyAsync(h.data(), cu.data, h.size() * sizeof(std::int32_t), cudaMemcpyDeviceToHost, stream));
  MI_CUDA_CHECK(cudaStreamSynchronize(stream));
  if (h.front() != 0) fail("cu[0] = " + std::to_string(h.front()) + ", expected 0");
  std::int64_t longest = 0;
  for (std::size_t b = 0; b + 1 < h.size(); ++b) {
    if (h[b + 1] < h[b]) fail("not non-decreasing at sequence " + std::to_string(b));
    longest = std::max<std::int64_t>(longest, static_cast<std::int64_t>(h[b + 1]) - h[b]);
  }
  if (h.back() != expected_tokens) {
    fail("cu[N] = " + std::to_string(h.back()) + " but the tensors have " + std::to_string(expected_tokens) + " rows");
  }
  if (longest > max_seqlen) {
    fail("max_seqlen = " + std::to_string(max_seqlen) + " is smaller than the longest sequence (" +
         std::to_string(longest) + "); the tail of longer sequences would be silently dropped");
  }
}

}  // namespace mi
