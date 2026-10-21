// MossInfer addition (not upstream): replaces the two ATen headers FA2 includes for its dropout RNG
// state (ATen/cuda/CUDAGeneratorImpl.h and ATen/cuda/detail/UnpackRaw.cuh). MossInfer never enables
// dropout, so the state is an inert value type and the build needs no PyTorch.
#pragma once

#include <cstdint>
#include <tuple>

namespace at {

struct PhiloxCudaState {
    uint64_t seed = 0;
    uint64_t offset = 0;
};

namespace cuda::philox {
__host__ __device__ __forceinline__ std::tuple<uint64_t, uint64_t> unpack(const PhiloxCudaState &arg) {
    return std::make_tuple(arg.seed, arg.offset);
}
}  // namespace cuda::philox

}  // namespace at
