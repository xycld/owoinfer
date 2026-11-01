// The VAE's single-head attention uses fp32 throughout. A bounded score slab lets cuBLASLt reuse Q/K/V tiles,
// replacing the compatibility kernel's one-dot-product-per-warp loop without narrowing its fp32 operands.
#include "kernels/sm70/vae_sm70.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "core/cuda_check.h"

namespace mi::kernels::sm70 {
namespace {

constexpr std::size_t kScoresBytes = std::size_t{64} << 20;
constexpr std::size_t kLtBytes = std::size_t{8} << 20;
constexpr int kThreads = 256;

int slab_rows(std::int64_t length) {
  if (length <= 0) return 0;
  return static_cast<int>(std::min<std::int64_t>(length, std::max<std::int64_t>(1,
      std::min<std::int64_t>(2048, kScoresBytes / (static_cast<std::size_t>(length) * sizeof(float))))));
}

std::size_t score_bytes(std::int64_t length) {
  const auto bytes = static_cast<std::size_t>(slab_rows(length)) * static_cast<std::size_t>(length) * sizeof(float);
  return (bytes + 255) / 256 * 256;
}

__global__ void softmax_kernel(float* scores, int columns) {
  __shared__ float partial[8];
  float* row = scores + static_cast<std::int64_t>(blockIdx.x) * columns;
  const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
  float value = -INFINITY;
  for (int c = tid; c < columns; c += kThreads) value = fmaxf(value, row[c]);
#pragma unroll
  for (int d = 16; d > 0; d >>= 1) value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, d));
  if (lane == 0) partial[warp] = value;
  __syncthreads();
  value = lane < 8 ? partial[lane] : -INFINITY;
#pragma unroll
  for (int d = 16; d > 0; d >>= 1) value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, d));
  const float maximum = value;
  __syncthreads();
  float sum = 0.f;
  for (int c = tid; c < columns; c += kThreads) {
    const float p = expf(row[c] - maximum);
    row[c] = p;
    sum += p;
  }
#pragma unroll
  for (int d = 16; d > 0; d >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, d);
  if (lane == 0) partial[warp] = sum;
  __syncthreads();
  sum = lane < 8 ? partial[lane] : 0.f;
#pragma unroll
  for (int d = 16; d > 0; d >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, d);
  const float inv = 1.f / sum;
  for (int c = tid; c < columns; c += kThreads) row[c] *= inv;
}

void check_lt(cublasStatus_t status, const char* what) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string("sm70::vae_attention_fp32: cuBLASLt ") + what + " failed: " +
                             cublasLtGetStatusString(status));
  }
}

// Column-major C[n,m] = op(A) * B. Plans live only for this call, which also avoids a device/stream cache or lock.
struct Matmul {
  cublasLtMatmulDesc_t desc = nullptr;
  cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
  cublasLtMatmulPreference_t pref = nullptr;
  cublasLtMatmulAlgo_t algo{};

  Matmul() = default;
  Matmul(const Matmul&) = delete;
  Matmul& operator=(const Matmul&) = delete;
  ~Matmul() {
    if (pref) cublasLtMatmulPreferenceDestroy(pref);
    if (c) cublasLtMatrixLayoutDestroy(c);
    if (b) cublasLtMatrixLayoutDestroy(b);
    if (a) cublasLtMatrixLayoutDestroy(a);
    if (desc) cublasLtMatmulDescDestroy(desc);
  }

  void init(cublasLtHandle_t lt, int m, int n, int k, std::int64_t lda, std::int64_t ldb, std::int64_t ldc,
            bool transpose_a) {
    check_lt(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F), "descriptor");
    const cublasOperation_t op = transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    check_lt(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &op, sizeof(op)), "transpose");
    check_lt(cublasLtMatrixLayoutCreate(&a, CUDA_R_32F, transpose_a ? k : n, transpose_a ? n : k, lda), "A layout");
    check_lt(cublasLtMatrixLayoutCreate(&b, CUDA_R_32F, k, m, ldb), "B layout");
    check_lt(cublasLtMatrixLayoutCreate(&c, CUDA_R_32F, n, m, ldc), "C layout");
    check_lt(cublasLtMatmulPreferenceCreate(&pref), "preference");
    check_lt(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &kLtBytes,
                                                 sizeof(kLtBytes)), "workspace");
    cublasLtMatmulHeuristicResult_t result{};
    int found = 0;
    check_lt(cublasLtMatmulAlgoGetHeuristic(lt, desc, a, b, c, c, pref, 1, &result, &found), "heuristic");
    if (found == 0) throw std::runtime_error("sm70::vae_attention_fp32: no fp32 GEMM algorithm");
    algo = result.algo;
  }

  void run(cublasLtHandle_t lt, const float* ap, const float* bp, float* cp, float alpha, void* workspace,
           cudaStream_t stream) const {
    const float beta = 0.f;
    check_lt(cublasLtMatmul(lt, desc, &alpha, ap, a, bp, b, &beta, cp, c, cp, c, &algo, workspace, kLtBytes, stream),
              "matmul");
  }
};

}  // namespace

std::size_t vae_attention_workspace_bytes(std::int64_t length) {
  // mid_attention accepts a 16-byte-aligned base; reserve room to align the cuBLASLt scratch to 256 bytes.
  return length > 0 ? score_bytes(length) + kLtBytes + 256 : 0;
}

void vae_attention_fp32(cublasLtHandle_t lt, const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& out,
                        float scale, void* workspace, std::size_t workspace_bytes, cudaStream_t stream) {
  const int length = static_cast<int>(q.dim(1)), dim = static_cast<int>(q.dim(2));
  if (length == 0 || q.dim(0) == 0) return;
  if (workspace == nullptr || workspace_bytes < vae_attention_workspace_bytes(length) ||
      reinterpret_cast<std::uintptr_t>(workspace) % 16 != 0) {
    throw std::invalid_argument("sm70::vae_attention_fp32: workspace too small or not 16-byte aligned");
  }
  const auto padding = (256 - reinterpret_cast<std::uintptr_t>(workspace) % 256) % 256;
  auto* aligned = static_cast<char*>(workspace) + padding;
  auto* scores = reinterpret_cast<float*>(aligned);
  void* scratch = aligned + score_bytes(length);
  const int chunk = slab_rows(length), tail = length % chunk;
  Matmul qk, pv, tail_qk, tail_pv;
  qk.init(lt, chunk, length, dim, k.stride[1], q.stride[1], length, true);
  pv.init(lt, chunk, dim, length, v.stride[1], length, out.stride[1], false);
  if (tail != 0) {
    tail_qk.init(lt, tail, length, dim, k.stride[1], q.stride[1], length, true);
    tail_pv.init(lt, tail, dim, length, v.stride[1], length, out.stride[1], false);
  }
  for (std::int64_t batch = 0; batch < q.dim(0); ++batch) {
    for (int start = 0; start < length; start += chunk) {
      const int rows = std::min(chunk, length - start);
      const bool full = rows == chunk;
      (full ? qk : tail_qk).run(lt, k.ptr<float>() + batch * k.stride[0],
                               q.ptr<float>() + batch * q.stride[0] + start * q.stride[1], scores, scale, scratch, stream);
      softmax_kernel<<<rows, kThreads, 0, stream>>>(scores, length);
      MI_CUDA_CHECK_LAUNCH();
      (full ? pv : tail_pv).run(lt, v.ptr<float>() + batch * v.stride[0], scores,
                               out.ptr<float>() + batch * out.stride[0] + start * out.stride[1], 1.f, scratch, stream);
    }
  }
}

}  // namespace mi::kernels::sm70
