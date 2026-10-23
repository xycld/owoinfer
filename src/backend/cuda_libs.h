// NVIDIA cuBLASLt, loaded at run time (dlopen / LoadLibraryExW), never linked: by default the copy bundled next to
// the engine, a compatible system copy (a CUDA install) when the package has none (the system-CUDA variant) or when
// asked for, and the engine still loads, and the probe still answers, when no copy is usable. Every cublasLt* call in the engine goes through the shim in
// cublaslt_shim.cpp, which forwards to the entry points resolved here.
//
// Which copy (docs/integration.md, "CUDA libraries"):
//   MOSSINFER_CUBLASLT=<file>    that file only (source "env");
//   MOSSINFER_CUDA_LIBS=auto     (default, also when empty) the bundled candidates, then the system ones;
//                       system   the system candidates only;
//                       bundled  the bundled candidates only.
// A copy is usable when cublasLtGetVersion() has this build's major version and is not older than the cuBLASLt the
// build was compiled against (CUBLAS_VERSION), and every entry point the shim forwards to resolves.
#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace mi::backend {

// No usable cuBLASLt (mi_engine_create: MI_ERR_CUDA). The message lists every candidate tried and why it was rejected.
struct CudaLibraryError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

enum class CudaLibPolicy { kAuto, kSystem, kBundled };

// MOSSINFER_CUDA_LIBS: "auto" (also ""), "system" or "bundled"; anything else throws CudaLibraryError.
CudaLibPolicy parse_cuda_lib_policy(std::string_view value);
const char* cuda_lib_policy_name(CudaLibPolicy policy);

// cublasLtGetVersion() encoding: major * 10000 + minor * 100 + patch. Same major, and runtime >= build.
bool cublaslt_compatible(std::size_t runtime_version, std::size_t build_version);
// "13.8.0" for 130800.
std::string cublaslt_version_string(std::size_t version);
// CUBLAS_VERSION of the headers this build was compiled against.
std::size_t cublaslt_build_version();

enum class LibOs { kLinux, kWindows };

// Environment variables as UTF-8 name/value pairs, injected so that the candidate lists are testable.
using Environment = std::vector<std::pair<std::string, std::string>>;
Environment process_environment();
// The value of `name` (case-insensitive on Windows); empty when unset or set to "".
std::string env_value(const Environment& env, std::string_view name, LibOs os);

struct Candidate {
  // An absolute file path, or (Linux, source "system") the plain soname "libcublasLt.so.<major>" for the dynamic
  // loader's default search (LD_LIBRARY_PATH, the ld.so cache).
  std::string path;
  std::string source;  // "env", "system" or "bundled"
  bool operator==(const Candidate&) const = default;
};

// The candidates in the order they are tried, de-duplicated. `module_dir`: the directory of the module that contains
// the engine (libmossinfer / mossinfer.dll, or an executable that links the core statically); "" leaves out the
// bundled candidates. Pure.
std::vector<Candidate> cublaslt_candidates(CudaLibPolicy policy, const Environment& env, const std::string& module_dir,
                                           LibOs os, int cuda_major);
// The bundled candidates alone (what `bundled` tries), for recognising a system search that found the bundled copy.
std::vector<Candidate> cublaslt_bundled_candidates(const std::string& module_dir, LibOs os, int cuda_major);

// The directory of the module this code is linked into, UTF-8; "" if unknown.
std::string own_module_dir();
LibOs current_os();

struct CublasLtInfo {
  std::string path;     // the file loaded (resolved, UTF-8)
  std::string source;   // "env", "system" or "bundled"
  std::size_t version = 0;  // cublasLtGetVersion()
};

struct CublasLtStatus {
  bool ok = false;
  CublasLtInfo info;   // when ok
  std::string error;   // when not ok: every candidate tried and why it was rejected
};

// The copy the process uses: loaded once (thread-safe) on the first call of either function or of any shim entry
// point, from the process environment, and kept loaded for the life of the process.
const CublasLtInfo& load_cublaslt();  // throws CudaLibraryError
CublasLtStatus cublaslt_status() noexcept;

// What the process-wide search would pick with this policy and environment, without making it the process's copy
// (tests and diagnostics). A usable copy stays loaded.
CublasLtStatus find_cublaslt(CudaLibPolicy policy, const Environment& env);
// Tries `candidates` in order, as the search does: the first usable one stays loaded, every rejected one is closed
// again. A "system" candidate that is the same file as one of `bundled` is rejected. For tests.
CublasLtStatus try_cublaslt_candidates(const std::vector<Candidate>& candidates, const std::vector<Candidate>& bundled);

// The entry points the shim forwards to, in this order (cublaslt_shim.cpp must define every one of them).
//
// The shim's definitions must never become dynamic symbols: libcublasLt's own internal calls to cublasLt* functions
// could then bind to the shim (symbol interposition) instead of to the library itself, and recurse or call into an
// uninitialised table. They are hidden in libmossinfer (visibility preset and version script), and executables that
// link the core statically must not export their symbols: cmake/modules/backend.cmake fails the configuration on
// ENABLE_EXPORTS, -rdynamic / --export-dynamic or a cuBLASLt link on any target, and check_package.py checks the
// shipped binaries.
#define MI_CUBLASLT_SYMBOLS(X)                                                                                      \
  X(cublasLtCreate) X(cublasLtDestroy) X(cublasLtGetVersion) X(cublasLtGetStatusString) X(cublasLtMatmul)          \
  X(cublasLtMatmulDescCreate) X(cublasLtMatmulDescDestroy) X(cublasLtMatmulDescSetAttribute)                      \
  X(cublasLtMatrixLayoutCreate) X(cublasLtMatrixLayoutDestroy) X(cublasLtMatmulPreferenceCreate)                  \
  X(cublasLtMatmulPreferenceDestroy) X(cublasLtMatmulPreferenceSetAttribute) X(cublasLtMatmulAlgoGetHeuristic)    \
  X(cublasLtMatmulAlgoConfigGetAttribute)

#define MI_CUBLASLT_ENUM(name) k_##name,
enum CublasLtEntry : int { MI_CUBLASLT_SYMBOLS(MI_CUBLASLT_ENUM) kCublasLtEntryCount };
#undef MI_CUBLASLT_ENUM

// The resolved entry points of the process's copy, indexed by CublasLtEntry; nullptr when no copy is usable.
void* const* cublaslt_entry_points() noexcept;
// Why no copy is usable, as a static string ("" when one is).
const char* cublaslt_unavailable_message() noexcept;

}  // namespace mi::backend
