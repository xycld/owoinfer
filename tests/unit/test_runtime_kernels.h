// Tiny kernels for runtime tests (compiled by nvcc in test_runtime_kernels.cu).
#pragma once

#include <cuda_runtime.h>

namespace mi::testing {

void launch_add(int* data, int n, int delta, cudaStream_t s);
// Busy-waits on the device for `ns` nanoseconds (globaltimer), occupying one SM.
void launch_spin(unsigned long long ns, cudaStream_t s);
// dst[i] = src[i] + 1 for one int.
void launch_copy_plus_one(const int* src, int* dst, cudaStream_t s);

}  // namespace mi::testing
