/* Test-only stand-in for cuBLASLt (cmake/modules/backend.cmake builds it four times): it exports the entry points the
 * engine's loader resolves (backend/cuda_libs.h, MI_CUBLASLT_SYMBOLS) and a cublasLtGetVersion() chosen per build,
 * so CudaLibs.* can drive every branch of the loader without NVIDIA's library:
 *   MI_FAKE_GOOD   the build's own CUBLAS_VERSION: usable
 *   MI_FAKE_OLD    one patch level older (0 when the build's version is N.0.0, which has no older same-major version)
 *   MI_FAKE_MAJOR  the next major version
 *   MI_FAKE_NOSYM  the build's version, without cublasLtMatmulAlgoConfigGetAttribute
 * Only the symbols matter: nothing calls the stubs. */
/* CUBLAS_VERSION without cublasLt.h's declarations, which these stubs would contradict. CUDA 12's cublas_api.h refuses
 * to be included unless CUBLASAPI is defined; cublasLt.h defines it empty in the same way before including it. */
#define CUBLASAPI
#include <cublas_api.h>

#include <stddef.h>

#ifdef _WIN32
#define MI_FAKE_EXPORT __declspec(dllexport)
#else
#define MI_FAKE_EXPORT __attribute__((visibility("default")))
#endif

#if defined(MI_FAKE_OLD)
#define MI_FAKE_VERSION (CUBLAS_VERSION % 10000 == 0 ? 0 : CUBLAS_VERSION - 1)
#elif defined(MI_FAKE_MAJOR)
#define MI_FAKE_VERSION (CUBLAS_VERSION + 10000)
#else
#define MI_FAKE_VERSION CUBLAS_VERSION
#endif

MI_FAKE_EXPORT size_t cublasLtGetVersion(void) { return (size_t)MI_FAKE_VERSION; }

#define MI_FAKE_STUB(name) \
  MI_FAKE_EXPORT int name(void) { return 0; }

MI_FAKE_STUB(cublasLtCreate)
MI_FAKE_STUB(cublasLtDestroy)
MI_FAKE_STUB(cublasLtGetStatusString)
MI_FAKE_STUB(cublasLtMatmul)
MI_FAKE_STUB(cublasLtMatmulDescCreate)
MI_FAKE_STUB(cublasLtMatmulDescDestroy)
MI_FAKE_STUB(cublasLtMatmulDescSetAttribute)
MI_FAKE_STUB(cublasLtMatrixLayoutCreate)
MI_FAKE_STUB(cublasLtMatrixLayoutDestroy)
MI_FAKE_STUB(cublasLtMatmulPreferenceCreate)
MI_FAKE_STUB(cublasLtMatmulPreferenceDestroy)
MI_FAKE_STUB(cublasLtMatmulPreferenceSetAttribute)
MI_FAKE_STUB(cublasLtMatmulAlgoGetHeuristic)
#ifndef MI_FAKE_NOSYM
MI_FAKE_STUB(cublasLtMatmulAlgoConfigGetAttribute)
#endif
