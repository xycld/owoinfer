// Device merge of the runtime LoRA stack's low-rank adapters into bf16 weights (kernels/lora/lora_merge.h): the same
// bits as merge_bf16 (lora.h), computed on the GPU where the weight already is, so a merge at load costs no host round
// trip (download, host merge, upload) and no host merge time.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string_view>

#include "models/lora/lora.h"

namespace mi::lora {

// Whether merge_bf16_device takes `module`: every adapter merge_bf16 would add to it is low rank without a bias delta
// (LoHa, LoKr, full deltas and bias deltas stay on the host merge). True when no adapter would be added.
bool device_mergeable(const LoraStack& stack, std::string_view module);

// Device scratch merge_bf16_device needs at least for `module` (its rows then go one row tile at a time); more scratch
// means fewer, larger row chunks. host_weight: the weight is in pinned host memory (its row chunks are staged in the
// scratch too).
std::size_t device_merge_min_scratch(const LoraStack& stack, std::string_view module, bool host_weight);
// Scratch that merges `module` in one chunk.
std::size_t device_merge_full_scratch(const LoraStack& stack, std::string_view module, bool host_weight);
// Rows merge_bf16_device takes per chunk with `scratch_bytes` of scratch (all of the module's rows from
// device_merge_full_scratch on; 0 below device_merge_min_scratch).
std::int64_t device_merge_chunk_rows(const LoraStack& stack, std::string_view module, bool host_weight,
                                     std::size_t scratch_bytes);

// merge_bf16 of `module` into the bf16 weight w [n, k] (rows contiguous) on the GPU, bit for bit: w in device memory
// (merged in place), or with host_weight in pinned host memory (row chunks uploaded, merged and downloaded back). The
// module must be device_mergeable. scratch: device memory, 256-byte aligned, at least device_merge_min_scratch bytes.
// Runs on `stream` and synchronizes it. Returns the number of adapters merged (0: w untouched). Throws LoraError for a
// module that is not device_mergeable, a shape mismatch, too little scratch or a non-finite merged value (w is then
// partially merged and must be discarded, as with merge_bf16); weights::FormatError for a non-finite factor.
// The module's bias is not touched: with no bias delta merge_bf16 only widens it to fp32 and rounds it back, which
// gives the same bits for every finite bias (merge_bf16 also refuses a non-finite bias; this path never reads it).
int merge_bf16_device(const LoraStack& stack, std::string_view module, std::uint16_t* w, std::int64_t n, std::int64_t k,
                      bool host_weight, void* scratch, std::size_t scratch_bytes, cudaStream_t stream);

}  // namespace mi::lora
