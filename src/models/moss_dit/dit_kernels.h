// Small CUDA helpers of the DiT: latent <-> token layout at the model boundary (P:247-248, P:259)
// and host-to-device writes of tiny per-call parameters.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>

#include "core/tensor.h"

namespace mi::moss_dit {

// hidden_states.flatten(2).transpose(1, 2).to(bf16): latents fp32 [C, h, w] (contiguous) -> tokens
// bf16 [h*w, C] (contiguous), round to nearest even.
void latent_to_tokens(const Tensor& latents, const Tensor& tokens, cudaStream_t stream);

// x.transpose(1, 2).reshape(C, h, w).float(): tokens bf16 [h*w, C] (contiguous) -> fp32 [C, h, w] (contiguous).
void tokens_to_latent(const Tensor& tokens, const Tensor& latents, cudaStream_t stream);

// Copies `bytes` (<= kMaxParamBytes) host bytes to dst through kernel arguments: no pinned staging,
// no host synchronization (a pageable cudaMemcpyAsync would wait for the stream), graph-capturable.
// dst: device, 4-byte aligned; bytes a multiple of 4.
inline constexpr std::size_t kMaxParamBytes = 2048;
void write_params(void* dst, const void* src, std::size_t bytes, cudaStream_t stream);

}  // namespace mi::moss_dit
