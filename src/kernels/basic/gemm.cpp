#include "kernels/basic/gemm.h"

#include <cublasLt.h>

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>

#include "core/cuda_check.h"
#include "kernels/basic/detail.h"

namespace mi::kernels {
namespace {

void check_lt(cublasStatus_t status, const char* expr) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw CudaError(std::string("cuBLASLt error ") + cublasLtGetStatusString(status) + " in " + expr);
  }
}
#define MI_LT_CHECK(expr) check_lt((expr), #expr)

// Largest power of two (<= 256) dividing the address: what cuBLASLt may assume about a buffer.
std::uint32_t alignment(const void* p) {
  const auto a = reinterpret_cast<std::uintptr_t>(p);
  std::uint32_t align = 256;
  while (align > 1 && a % align != 0) align /= 2;
  return align;
}

// Single row stride of a [..., C] view whose leading dims flatten to M rows; 0 if they do not.
std::int64_t row_stride(const Tensor& t) {
  const detail::RowMap m = detail::row_map(t);
  if (m.nd == 0) return t.dim(-1);
  return m.nd == 1 ? m.stride[0] : 0;
}

cudaDataType_t cuda_type(DType t) { return t == DType::kF32 ? CUDA_R_32F : CUDA_R_16BF; }

struct Key {
  std::int64_t m, n, k, ldx, ldw, ldo;
  int out_type, bias_type;  // bias_type -1: no bias
  std::size_t workspace;
  std::uint32_t align_x, align_w, align_out, align_bias;
  bool operator==(const Key&) const = default;
};

struct KeyHash {
  std::size_t operator()(const Key& key) const {
    std::size_t h = 1469598103934665603ull;
    const auto mix = [&h](std::uint64_t v) { h = (h ^ v) * 1099511628211ull; };
    for (std::int64_t v : {key.m, key.n, key.k, key.ldx, key.ldw, key.ldo}) mix(static_cast<std::uint64_t>(v));
    for (std::uint64_t v : {std::uint64_t(key.out_type + 8), std::uint64_t(key.bias_type + 8), std::uint64_t(key.workspace),
                            std::uint64_t(key.align_x), std::uint64_t(key.align_w), std::uint64_t(key.align_out),
                            std::uint64_t(key.align_bias)}) {
      mix(v);
    }
    return h;
  }
};

// Row-major out[M,N] = x[M,K]·w[N,K]ᵀ, read column-major, is outᵀ = op_T(w)·x: A = w (stored K x N, op T),
// B = x (stored K x M, op N), D = out (N x M), so cuBLASLt m = N, n = M (torch's "TN" layout).
struct Plan {
  cublasLtMatmulDesc_t desc = nullptr;
  cublasLtMatrixLayout_t a = nullptr, b = nullptr, d = nullptr;
  cublasLtMatmulAlgo_t algo{};
  std::size_t workspace = 0;

  Plan() = default;
  Plan(const Plan&) = delete;
  Plan& operator=(const Plan&) = delete;
  ~Plan() {
    if (d != nullptr) cublasLtMatrixLayoutDestroy(d);
    if (b != nullptr) cublasLtMatrixLayoutDestroy(b);
    if (a != nullptr) cublasLtMatrixLayoutDestroy(a);
    if (desc != nullptr) cublasLtMatmulDescDestroy(desc);
  }
};

}  // namespace

struct Gemm::Impl {
  cublasLtHandle_t handle = nullptr;
  mutable std::mutex mutex;
  std::unordered_map<Key, std::unique_ptr<Plan>, KeyHash> plans;

  Plan& plan_for(const Key& key, const void* bias_ptr) {
    auto it = plans.find(key);
    if (it != plans.end()) return *it->second;
    auto plan = std::make_unique<Plan>();
    MI_LT_CHECK(cublasLtMatmulDescCreate(&plan->desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    const cublasOperation_t trans_a = CUBLAS_OP_T, trans_b = CUBLAS_OP_N;
    MI_LT_CHECK(cublasLtMatmulDescSetAttribute(plan->desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)));
    MI_LT_CHECK(cublasLtMatmulDescSetAttribute(plan->desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)));
    if (key.bias_type >= 0) {
      const cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
      const auto bias_type = static_cast<std::int32_t>(cuda_type(static_cast<DType>(key.bias_type)));
      MI_LT_CHECK(cublasLtMatmulDescSetAttribute(plan->desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
      MI_LT_CHECK(cublasLtMatmulDescSetAttribute(plan->desc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_type,
                                                 sizeof(bias_type)));
      // The heuristic reads the pointer's alignment (part of the key), never its contents.
      MI_LT_CHECK(cublasLtMatmulDescSetAttribute(plan->desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_ptr,
                                                 sizeof(bias_ptr)));
    }
    const auto u64 = [](std::int64_t v) { return static_cast<std::uint64_t>(v); };
    MI_LT_CHECK(cublasLtMatrixLayoutCreate(&plan->a, CUDA_R_16BF, u64(key.k), u64(key.n), key.ldw));
    MI_LT_CHECK(cublasLtMatrixLayoutCreate(&plan->b, CUDA_R_16BF, u64(key.k), u64(key.m), key.ldx));
    MI_LT_CHECK(cublasLtMatrixLayoutCreate(&plan->d, cuda_type(static_cast<DType>(key.out_type)), u64(key.n),
                                           u64(key.m), key.ldo));

    cublasLtMatmulPreference_t pref = nullptr;
    MI_LT_CHECK(cublasLtMatmulPreferenceCreate(&pref));
    struct PrefGuard {
      cublasLtMatmulPreference_t p;
      ~PrefGuard() { cublasLtMatmulPreferenceDestroy(p); }
    } guard{pref};
    const std::size_t ws = key.workspace;
    // Split-K partials are reduced in fp32 only, never in the bf16 output buffer.
    const std::uint32_t schemes = CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;
    MI_LT_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));
    MI_LT_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &schemes,
                                                     sizeof(schemes)));
    MI_LT_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, &key.align_w,
                                                     sizeof(key.align_w)));
    MI_LT_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, &key.align_x,
                                                     sizeof(key.align_x)));
    MI_LT_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, &key.align_out,
                                                     sizeof(key.align_out)));
    MI_LT_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, &key.align_out,
                                                     sizeof(key.align_out)));
    cublasLtMatmulHeuristicResult_t result{};
    int found = 0;
    MI_LT_CHECK(cublasLtMatmulAlgoGetHeuristic(handle, plan->desc, plan->a, plan->b, plan->d, plan->d, pref, 1, &result,
                                               &found));
    if (found == 0) {
      throw CudaError("cuBLASLt: no algorithm for linear M=" + std::to_string(key.m) + " N=" + std::to_string(key.n) +
                      " K=" + std::to_string(key.k));
    }
    plan->algo = result.algo;
    plan->workspace = result.workspaceSize;
    return *plans.emplace(key, std::move(plan)).first->second;
  }
};

Gemm::Gemm() : impl_(std::make_unique<Impl>()) { MI_LT_CHECK(cublasLtCreate(&impl_->handle)); }

Gemm::~Gemm() {
  impl_->plans.clear();
  if (impl_->handle != nullptr) cublasLtDestroy(impl_->handle);
}

std::size_t Gemm::cached_plans() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->plans.size();
}

namespace {

// Validates linear()'s operands and returns their plan key; m or n of 0 means there is nothing to compute.
Key linear_key(const Tensor& out, const Tensor& x, const Tensor& w, const Tensor* bias, std::size_t workspace_bytes) {
  constexpr const char* op = "linear";
  detail::check_rows(op, "x", x, DType::kBF16);
  const std::int64_t k = x.dim(-1);
  detail::check_rows(op, "w", w, DType::kBF16, k);
  if (w.rank != 2) detail::fail(op, "w must be [N, K], got " + w.describe());
  const std::int64_t n = w.dim(0);
  if (out.dtype != DType::kBF16 && out.dtype != DType::kF32) detail::fail(op, "out must be bf16 or f32, got " + out.describe());
  detail::check_rows(op, "out", out, out.dtype, n);
  const std::int64_t m = detail::rows_of(x);
  if (detail::rows_of(out) != m) detail::fail(op, "row count mismatch: " + x.describe() + " vs " + out.describe());
  const std::int64_t ldx = row_stride(x), ldo = row_stride(out), ldw = row_stride(w);
  if (ldx < k || ldo < n || ldw < k) {
    detail::fail(op, "x, w and out must each be rows with one stride: " + x.describe() + ", " + w.describe() + ", " +
                         out.describe());
  }
  int bias_type = -1;
  if (bias != nullptr) {
    // cuBLASLt's bias epilogue takes a bias of the output type only (bf16 bias with f32 out is rejected).
    detail::check_rows(op, "bias", *bias, out.dtype, n);
    if (bias->numel() != n) detail::fail(op, "bias must be [N], got " + bias->describe());
    bias_type = static_cast<int>(bias->dtype);
  }
  const void* bias_ptr = bias != nullptr ? bias->data : nullptr;
  return Key{m, n, k, ldx, ldw, ldo, static_cast<int>(out.dtype), bias_type, workspace_bytes,
             alignment(x.data), alignment(w.data), alignment(out.data), bias_ptr ? alignment(bias_ptr) : 0u};
}

template <class T>
T algo_attribute(const cublasLtMatmulAlgo_t& algo, cublasLtMatmulAlgoConfigAttributes_t attr) {
  T v{};
  std::size_t written = 0;
  MI_LT_CHECK(cublasLtMatmulAlgoConfigGetAttribute(&algo, attr, &v, sizeof(v), &written));
  return v;
}

}  // namespace

std::string Gemm::AlgoConfig::describe() const {
  return "algo " + std::to_string(id) + " tile " + std::to_string(tile) + " stages " + std::to_string(stages) +
         " split_k " + std::to_string(split_k) + " reduction " + std::to_string(reduction_scheme) + " swizzle " +
         std::to_string(swizzle) + " custom " + std::to_string(custom_option) + " inner " + std::to_string(inner_shape) +
         " cluster " + std::to_string(cluster_shape) + " workspace " + std::to_string(workspace_bytes);
}

Gemm::AlgoConfig Gemm::algo_config(const Tensor& out, const Tensor& x, const Tensor& w, const Tensor* bias,
                                   std::size_t workspace_bytes) {
  const Key key = linear_key(out, x, w, bias, workspace_bytes);
  if (key.m == 0 || key.n == 0) return {};
  std::lock_guard<std::mutex> lock(impl_->mutex);
  const Plan& plan = impl_->plan_for(key, bias != nullptr ? bias->data : nullptr);
  AlgoConfig c;
  c.id = algo_attribute<std::int32_t>(plan.algo, CUBLASLT_ALGO_CONFIG_ID);
  c.tile = algo_attribute<std::uint32_t>(plan.algo, CUBLASLT_ALGO_CONFIG_TILE_ID);
  c.split_k = algo_attribute<std::int32_t>(plan.algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM);
  c.reduction_scheme = algo_attribute<std::uint32_t>(plan.algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME);
  c.swizzle = algo_attribute<std::uint32_t>(plan.algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING);
  c.custom_option = algo_attribute<std::uint32_t>(plan.algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION);
  c.stages = algo_attribute<std::uint32_t>(plan.algo, CUBLASLT_ALGO_CONFIG_STAGES_ID);
  c.inner_shape = algo_attribute<std::uint16_t>(plan.algo, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID);
  c.cluster_shape = algo_attribute<std::uint16_t>(plan.algo, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID);
  c.workspace_bytes = plan.workspace;
  return c;
}

void Gemm::linear(const Tensor& out, const Tensor& x, const Tensor& w, const Tensor* bias, void* workspace,
                  std::size_t workspace_bytes, cudaStream_t stream) {
  const Key key = linear_key(out, x, w, bias, workspace_bytes);
  if (workspace_bytes > 0 && (workspace == nullptr || alignment(workspace) < 256)) {
    detail::fail("linear", "workspace must be 256-byte aligned");
  }
  if (key.m == 0 || key.n == 0) return;

  const void* bias_ptr = bias != nullptr ? bias->data : nullptr;
  std::lock_guard<std::mutex> lock(impl_->mutex);
  Plan& plan = impl_->plan_for(key, bias_ptr);
  if (bias_ptr != nullptr) {
    MI_LT_CHECK(cublasLtMatmulDescSetAttribute(plan.desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_ptr, sizeof(bias_ptr)));
  }
  const float alpha = 1.0f, beta = 0.0f;
  MI_LT_CHECK(cublasLtMatmul(impl_->handle, plan.desc, &alpha, w.data, plan.a, x.data, plan.b, &beta, out.data, plan.d,
                             out.data, plan.d, &plan.algo, workspace, plan.workspace, stream));
}

}  // namespace mi::kernels
