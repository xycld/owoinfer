// ================================================================================================================
// cuBLASLt shim. The engine does not link cuBLASLt: every cublasLt* function it calls is defined HERE, with the exact
// signature of <cublasLt.h>, and forwards to the copy backend/cuda_libs.cpp loaded at run time.
//
// ADDING A NEW cublasLt* CALL ANYWHERE (src/, tests/) REQUIRES AN ENTRY HERE AND IN MI_CUBLASLT_SYMBOLS
// (backend/cuda_libs.h). Until it has one, the link fails with an undefined reference to that function.
//
// Without a usable copy the functions never throw (they are C functions): status functions return
// CUBLAS_STATUS_NOT_INITIALIZED, cublasLtGetVersion() returns 0 and cublasLtGetStatusString() a static string naming
// the load failure. In the shared library these definitions are hidden (and outside the version script): they never
// leave libmossinfer, and libmossinfer has no DT_NEEDED / import entry for cuBLASLt. They must stay that way in every
// binary: exported, they could capture libcublasLt's own internal calls to these functions (backend/cuda_libs.h
// explains, cmake/modules/backend.cmake enforces it at configure time).
// ================================================================================================================
#include <cublasLt.h>

#include "backend/cuda_libs.h"

namespace {

// The loaded entry point `e` as the type of the declaration `F`; nullptr without a usable copy.
template <class F>
F entry(mi::backend::CublasLtEntry e) noexcept {
  void* const* table = mi::backend::cublaslt_entry_points();
  return table != nullptr ? reinterpret_cast<F>(table[e]) : nullptr;
}

#define MI_LT_FORWARD(name, ...)                                                       \
  const auto f = entry<decltype(&::name)>(mi::backend::k_##name);                      \
  return f != nullptr ? f(__VA_ARGS__) : CUBLAS_STATUS_NOT_INITIALIZED

}  // namespace

extern "C" {

cublasStatus_t CUBLASWINAPI cublasLtCreate(cublasLtHandle_t* lightHandle) {
  MI_LT_FORWARD(cublasLtCreate, lightHandle);
}

cublasStatus_t CUBLASWINAPI cublasLtDestroy(cublasLtHandle_t lightHandle) {
  MI_LT_FORWARD(cublasLtDestroy, lightHandle);
}

size_t CUBLASWINAPI cublasLtGetVersion(void) {
  const auto f = entry<decltype(&::cublasLtGetVersion)>(mi::backend::k_cublasLtGetVersion);
  return f != nullptr ? f() : 0;
}

const char* CUBLASWINAPI cublasLtGetStatusString(cublasStatus_t status) {
  const auto f = entry<decltype(&::cublasLtGetStatusString)>(mi::backend::k_cublasLtGetStatusString);
  return f != nullptr ? f(status) : mi::backend::cublaslt_unavailable_message();
}

cublasStatus_t CUBLASWINAPI cublasLtMatmul(cublasLtHandle_t lightHandle, cublasLtMatmulDesc_t computeDesc,
                                           const void* alpha, const void* A, cublasLtMatrixLayout_t Adesc,
                                           const void* B, cublasLtMatrixLayout_t Bdesc, const void* beta,
                                           const void* C, cublasLtMatrixLayout_t Cdesc, void* D,
                                           cublasLtMatrixLayout_t Ddesc, const cublasLtMatmulAlgo_t* algo,
                                           void* workspace, size_t workspaceSizeInBytes, cudaStream_t stream) {
  MI_LT_FORWARD(cublasLtMatmul, lightHandle, computeDesc, alpha, A, Adesc, B, Bdesc, beta, C, Cdesc, D, Ddesc, algo,
                workspace, workspaceSizeInBytes, stream);
}

cublasStatus_t CUBLASWINAPI cublasLtMatmulDescCreate(cublasLtMatmulDesc_t* matmulDesc, cublasComputeType_t computeType,
                                                     cudaDataType_t scaleType) {
  MI_LT_FORWARD(cublasLtMatmulDescCreate, matmulDesc, computeType, scaleType);
}

cublasStatus_t CUBLASWINAPI cublasLtMatmulDescDestroy(cublasLtMatmulDesc_t matmulDesc) {
  MI_LT_FORWARD(cublasLtMatmulDescDestroy, matmulDesc);
}

cublasStatus_t CUBLASWINAPI cublasLtMatmulDescSetAttribute(cublasLtMatmulDesc_t matmulDesc,
                                                           cublasLtMatmulDescAttributes_t attr, const void* buf,
                                                           size_t sizeInBytes) {
  MI_LT_FORWARD(cublasLtMatmulDescSetAttribute, matmulDesc, attr, buf, sizeInBytes);
}

cublasStatus_t CUBLASWINAPI cublasLtMatrixLayoutCreate(cublasLtMatrixLayout_t* matLayout, cudaDataType type,
                                                       uint64_t rows, uint64_t cols, int64_t ld) {
  MI_LT_FORWARD(cublasLtMatrixLayoutCreate, matLayout, type, rows, cols, ld);
}

cublasStatus_t CUBLASWINAPI cublasLtMatrixLayoutDestroy(cublasLtMatrixLayout_t matLayout) {
  MI_LT_FORWARD(cublasLtMatrixLayoutDestroy, matLayout);
}

cublasStatus_t CUBLASWINAPI cublasLtMatmulPreferenceCreate(cublasLtMatmulPreference_t* pref) {
  MI_LT_FORWARD(cublasLtMatmulPreferenceCreate, pref);
}

cublasStatus_t CUBLASWINAPI cublasLtMatmulPreferenceDestroy(cublasLtMatmulPreference_t pref) {
  MI_LT_FORWARD(cublasLtMatmulPreferenceDestroy, pref);
}

cublasStatus_t CUBLASWINAPI cublasLtMatmulPreferenceSetAttribute(cublasLtMatmulPreference_t pref,
                                                                 cublasLtMatmulPreferenceAttributes_t attr,
                                                                 const void* buf, size_t sizeInBytes) {
  MI_LT_FORWARD(cublasLtMatmulPreferenceSetAttribute, pref, attr, buf, sizeInBytes);
}

cublasStatus_t CUBLASWINAPI cublasLtMatmulAlgoGetHeuristic(cublasLtHandle_t lightHandle,
                                                           cublasLtMatmulDesc_t operationDesc,
                                                           cublasLtMatrixLayout_t Adesc, cublasLtMatrixLayout_t Bdesc,
                                                           cublasLtMatrixLayout_t Cdesc, cublasLtMatrixLayout_t Ddesc,
                                                           cublasLtMatmulPreference_t preference,
                                                           int requestedAlgoCount,
                                                           cublasLtMatmulHeuristicResult_t heuristicResultsArray[],
                                                           int* returnAlgoCount) {
  MI_LT_FORWARD(cublasLtMatmulAlgoGetHeuristic, lightHandle, operationDesc, Adesc, Bdesc, Cdesc, Ddesc, preference,
                requestedAlgoCount, heuristicResultsArray, returnAlgoCount);
}

cublasStatus_t CUBLASWINAPI cublasLtMatmulAlgoConfigGetAttribute(const cublasLtMatmulAlgo_t* algo,
                                                                 cublasLtMatmulAlgoConfigAttributes_t attr, void* buf,
                                                                 size_t sizeInBytes, size_t* sizeWritten) {
  MI_LT_FORWARD(cublasLtMatmulAlgoConfigGetAttribute, algo, attr, buf, sizeInBytes, sizeWritten);
}

}  // extern "C"
