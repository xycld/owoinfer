// Test seams of the weights module: internal (not part of any API, not exported from libmossinfer), for tests that
// must reach failure and fallback paths a healthy machine never takes. Every seam is off by default and process-wide.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace mi::weights::testing {

// Pinned host allocations (load_group_to_host, read_files_to_host) succeed until `bytes` more have been allocated,
// then each one throws CudaError as cudaHostAlloc's cudaErrorMemoryAllocation would. 0 fails the next one; reset with
// clear_fail_pinned_alloc().
void fail_pinned_alloc_after(std::uint64_t bytes);
void clear_fail_pinned_alloc();
// Called before every pinned allocation of `bytes`: throws when the seam says so.
void check_pinned_alloc(std::size_t bytes);

// Sets `*flag` to true after the first 64 MiB chunk read_files_to_host has read (once; the seam then disarms), so a
// test cancels a host copy midway. nullptr disarms. `flag` must outlive the copy.
void cancel_after_first_host_chunk(std::atomic<bool>* flag);
// Called after every chunk read_files_to_host reads: fires the seam above (one relaxed load when it is off).
void after_host_chunk();

// Sets `*flag` to true as the `nth` reference prefill from now starts (moss_dit::MossDit::prefill_references; 1 = the
// next; once, the seam then disarms), so a test cancels inside a given pass's prefill. nullptr disarms.
void cancel_at_reference_prefill(std::atomic<bool>* flag, int nth);
// Called as every reference prefill starts: fires the seam above (one relaxed load when it is off).
void before_reference_prefill();

// page_cache_fraction's query: kAuto (cachestat, else the mincore sample), kMincore (the sample alone, as on kernels
// without cachestat) or kNone (no query: unknown).
enum class ResidencyQuery { kAuto, kMincore, kNone };
void force_residency_query(ResidencyQuery query);
ResidencyQuery residency_query();

// The quantized importer (quant::open_quant_model) reads linears and text-encoder tensors of any shape instead of
// refusing every one that is not the MOSS-Image DiT's or text encoder's with its shape (weights::moss_dit_linears,
// weights::qwen35_text_tensors): for the reader's and converter's tests on small synthetic models. Off by default.
void allow_any_model_shapes(bool allow);
bool any_model_shapes();

}  // namespace mi::weights::testing
