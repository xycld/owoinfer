#include "core/cuda_check.h"
#include "test_runtime_kernels.h"

namespace mi::testing {

namespace {
__global__ void add_kernel(int* data, int n, int delta) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) data[i] += delta;
}
__global__ void spin_kernel(unsigned long long ns) {
  unsigned long long t0, t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
  do {
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  } while (t - t0 < ns);
}
__global__ void copy_plus_one_kernel(const int* src, int* dst) { *dst = *src + 1; }
}  // namespace

void launch_add(int* data, int n, int delta, cudaStream_t s) {
  add_kernel<<<(n + 255) / 256, 256, 0, s>>>(data, n, delta);
  MI_CUDA_CHECK_LAUNCH();
}
void launch_spin(unsigned long long ns, cudaStream_t s) {
  spin_kernel<<<1, 1, 0, s>>>(ns);
  MI_CUDA_CHECK_LAUNCH();
}
void launch_copy_plus_one(const int* src, int* dst, cudaStream_t s) {
  copy_plus_one_kernel<<<1, 1, 0, s>>>(src, dst);
  MI_CUDA_CHECK_LAUNCH();
}

}  // namespace mi::testing
