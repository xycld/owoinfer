// Device merge of low-rank adapters into a bf16 weight (models/lora merge_bf16_device): the host merge's arithmetic
// (models/lora merge_bf16, quant::lora_merge_row) on the GPU, so the merged bits are the same.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace mi::kernels::lora {

// w [n, k] (row stride ldw), in place, per element (i, c):
//   v = f32(w[i, c]);
//   per adapter a in order: acc = Σ_{j in [bounds[a], bounds[a + 1])} f64(up[i, j]) * f64(down[j, c]), fp64 with j
//     ascending (each product of two fp32 values is exact in fp64, so a fused multiply-add rounds as the host's
//     multiply then add); v = v + coef[a] * f32(acc) (fp32, each operation rounded on its own);
//   w[i, c] = bf16(v), round to nearest even.
// up fp32 [n, R] (row stride ldu), down fp32 [R, k] (row stride ldd), bounds int [adapters + 1] (ascending from 0 to
// R) and coef fp32 [adapters], all in device memory. A non-finite v sets *bad (device int) to 1; the element is
// written all the same (the caller discards the weight). No allocation, no synchronization. Throws
// std::invalid_argument on bad arguments.
void merge_low_rank_bf16(__nv_bfloat16* w, std::int64_t ldw, std::int64_t n, std::int64_t k, const float* up,
                         std::int64_t ldu, const float* down, std::int64_t ldd, const int* bounds, const float* coef,
                         int adapters, int* bad, cudaStream_t stream);

}  // namespace mi::kernels::lora
