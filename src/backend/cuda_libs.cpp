#include "backend/cuda_libs.h"

#include <cublasLt.h>  // never cublas_api.h directly: CUDA 12 requires CUBLASAPI, which cublasLt.h defines

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <cwchar>
#include <filesystem>
#include <new>
#include <system_error>

#include "util/path.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#include <link.h>
extern char** environ;
#endif

namespace fs = std::filesystem;

namespace mi::backend {
namespace {

constexpr const char* kSymbolNames[kCublasLtEntryCount] = {
#define MI_CUBLASLT_NAME(name) #name,
    MI_CUBLASLT_SYMBOLS(MI_CUBLASLT_NAME)
#undef MI_CUBLASLT_NAME
};

#if defined(__aarch64__) || defined(_M_ARM64)
constexpr const char* kLinuxTargetDir = "targets/sbsa-linux/lib";
constexpr const char* kMultiarchDir = "/usr/lib/aarch64-linux-gnu";
#else
constexpr const char* kLinuxTargetDir = "targets/x86_64-linux/lib";
constexpr const char* kMultiarchDir = "/usr/lib/x86_64-linux-gnu";
#endif

std::string lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

// `dir` + separator + `name`, without doubling a trailing separator of `dir`.
std::string join(std::string dir, std::string_view name, LibOs os) {
  const auto is_sep = [os](char c) { return c == '/' || (os == LibOs::kWindows && c == '\\'); };
  while (dir.size() > 1 && is_sep(dir.back())) dir.pop_back();
  return dir + (os == LibOs::kWindows ? "\\" : "/") + std::string(name);
}

std::string library_name(LibOs os, int cuda_major) {
  return os == LibOs::kWindows ? "cublasLt64_" + std::to_string(cuda_major) + ".dll"
                               : "libcublasLt.so." + std::to_string(cuda_major);
}

// The same path spelled once: case-insensitive on Windows, ignoring doubled separators, and on Linux lexically
// normalised ("/pkg/lib/../lib" is "/pkg/lib").
std::string path_key(const std::string& p, LibOs os) {
  if (os == LibOs::kLinux) return fs::path(p).lexically_normal().generic_string();
  std::string k;
  for (const char c : p) {
    const char n = c == '/' ? '\\' : c;
    if (!k.empty() && n == '\\' && k.back() == '\\') continue;
    k += n;
  }
  return lower(k);
}

// Appends unless the same path is already listed in `out` or in `exclude`.
void add(std::vector<Candidate>& out, std::string path, const char* source, LibOs os,
         const std::vector<Candidate>& exclude = {}) {
  const std::string k = path_key(path, os);
  for (const std::vector<Candidate>* list : {static_cast<const std::vector<Candidate>*>(&out), &exclude}) {
    for (const Candidate& c : *list) {
      if (path_key(c.path, os) == k) return;
    }
  }
  out.push_back(Candidate{std::move(path), source});
}

// "/x" on Linux; "C:\x", "C:/x" or a UNC "\\host\x" on Windows. Relative roots and files are never searched: they
// would depend on the process's working directory.
bool is_absolute(const std::string& p, LibOs os) {
  if (os == LibOs::kLinux) return !p.empty() && p[0] == '/';
  const auto sep = [](char c) { return c == '\\' || c == '/'; };
  return (p.size() >= 3 && std::isalpha(static_cast<unsigned char>(p[0])) && p[1] == ':' && sep(p[2])) ||
         (p.size() >= 2 && sep(p[0]) && sep(p[1]));
}

// The system candidates, minus any that name a bundled file (`bundled`): that file is the bundled copy.
void system_candidates(std::vector<Candidate>& out, const Environment& env, LibOs os, int cuda_major,
                       const std::vector<Candidate>& bundled) {
  const std::string lib = library_name(os, cuda_major);
  if (os == LibOs::kWindows) {
    // Never a bare-name load: the application directory comes first in the search and holds the bundled copy.
    std::vector<std::string> roots;
    if (std::string v = env_value(env, "CUDA_PATH", os); is_absolute(v, os)) roots.push_back(std::move(v));
    // CUDA_PATH_V<major>_<minor>, newest minor first.
    const std::string prefix = "CUDA_PATH_V" + std::to_string(cuda_major) + "_";
    std::vector<std::pair<int, std::string>> versioned;
    for (const auto& [name, value] : env) {
      if (value.empty() || name.size() <= prefix.size() || lower(name.substr(0, prefix.size())) != lower(prefix)) continue;
      int minor = -1;
      const char* first = name.data() + prefix.size();
      const char* last = name.data() + name.size();
      const auto [end, ec] = std::from_chars(first, last, minor);
      if (ec != std::errc() || end != last || !is_absolute(value, os)) continue;
      versioned.emplace_back(minor, value);
    }
    std::stable_sort(versioned.begin(), versioned.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    for (auto& [minor, value] : versioned) roots.push_back(std::move(value));
    for (const std::string& root : roots) {
      add(out, join(join(join(root, "bin", os), "x64", os), lib, os), "system", os, bundled);
      add(out, join(join(root, "bin", os), lib, os), "system", os, bundled);
    }
    return;
  }
  add(out, lib, "system", os);  // the dynamic loader's default search
  std::vector<std::string> roots;
  for (const char* var : {"CUDA_PATH", "CUDA_HOME"}) {
    if (std::string v = env_value(env, var, os); is_absolute(v, os)) roots.push_back(std::move(v));
  }
  roots.push_back("/usr/local/cuda-" + std::to_string(cuda_major));
  roots.push_back("/usr/local/cuda");
  for (const std::string& root : roots) {
    add(out, join(join(root, "lib64", os), lib, os), "system", os, bundled);
    add(out, join(join(root, kLinuxTargetDir, os), lib, os), "system", os, bundled);
  }
  // CUDA from a distribution's packages, by absolute path: the soname search above can miss them when the calling
  // module's RUNPATH (an executable's $ORIGIN/../lib) leads it to the bundled copy first.
  for (const char* dir : {"/opt/cuda/lib64", kMultiarchDir, "/usr/lib64", "/usr/lib"}) {
    add(out, join(dir, lib, os), "system", os, bundled);
  }
}

// ---------------------------------------------------------------- loading

#ifdef _WIN32
using Handle = HMODULE;
#else
using Handle = void*;
#endif

struct Opened {
  Handle handle = nullptr;
  std::string error;
};

Opened open_library(const Candidate& c) {
  Opened o;
#ifdef _WIN32
  fs::path native = util::path_from_utf8(c.path);
  native.make_preferred();  // LoadLibraryExW with LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR wants backslashes
  const std::wstring path = native.wstring();
  o.handle = LoadLibraryExW(path.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
  if (o.handle == nullptr) o.error = "not loadable (LoadLibraryExW error " + std::to_string(GetLastError()) + ")";
#else
  o.handle = dlopen(c.path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (o.handle == nullptr) {
    const char* e = dlerror();
    o.error = "not loadable (" + std::string(e != nullptr ? e : "dlopen failed") + ")";
  }
#endif
  return o;
}

void close_library(Handle h) {
#ifdef _WIN32
  FreeLibrary(h);
#else
  dlclose(h);
#endif
}

void* symbol(Handle h, const char* name) {
#ifdef _WIN32
  return reinterpret_cast<void*>(GetProcAddress(h, name));
#else
  return dlsym(h, name);
#endif
}

// The file a handle was loaded from (resolved through symbolic links where possible).
std::string loaded_path(Handle h, const std::string& fallback) {
  std::string file;
#ifdef _WIN32
  std::wstring buf(32768, L'\0');
  const DWORD n = GetModuleFileNameW(h, buf.data(), static_cast<DWORD>(buf.size()));
  if (n > 0 && n < buf.size()) {
    buf.resize(n);
    file = util::path_utf8(fs::path(buf));
  }
#else
  struct link_map* lm = nullptr;
  if (dlinfo(h, RTLD_DI_LINKMAP, &lm) == 0 && lm != nullptr && lm->l_name != nullptr && lm->l_name[0] != '\0') {
    file = lm->l_name;
  }
#endif
  if (file.empty()) file = fallback;
  std::error_code ec;
  const fs::path canonical = fs::canonical(util::path_from_utf8(file), ec);
  return ec ? file : util::path_utf8(canonical);
}

bool same_file(const std::string& a, const std::string& b) {
  std::error_code ec;
  return fs::equivalent(util::path_from_utf8(a), util::path_from_utf8(b), ec) && !ec;
}

bool has_dir(const std::string& path) { return path.find('/') != std::string::npos || path.find('\\') != std::string::npos; }

struct Attempt {
  CublasLtStatus status;
  std::array<void*, kCublasLtEntryCount> entry{};
};

using GetVersionFn = std::size_t (*)();

// The first usable candidate (kept loaded), else every rejection. `bundled` are the bundled files: a system candidate
// that resolves to one of them (a plain-name search through an executable's RUNPATH or LD_LIBRARY_PATH) is rejected,
// so that a copy reported as "system" never is the bundled one.
Attempt try_candidates(const std::vector<Candidate>& candidates, const std::vector<Candidate>& bundled,
                       const std::string& what) {
  Attempt a;
  const std::size_t build = cublaslt_build_version();
  std::string rejected;              // one line per candidate that exists but was rejected
  std::vector<std::string> missing;  // files that do not exist: one short line at the end
  for (const Candidate& c : candidates) {
    std::string reason;
    std::error_code ec;
    if (c.source == "env" && !is_absolute(c.path, current_os())) {
      reason = "not an absolute path (MOSSINFER_CUBLASLT names one file, never a search)";
    } else if (has_dir(c.path) && !fs::exists(util::path_from_utf8(c.path), ec)) {
      missing.push_back(c.path + " (" + c.source + ")");
      continue;
    } else {
      const Opened o = open_library(c);
      if (o.handle == nullptr) {
        reason = has_dir(c.path) ? o.error : "not found by the dynamic loader's search (" + o.error + ")";
      } else {
        const std::string file = loaded_path(o.handle, c.path);
        const auto get_version = reinterpret_cast<GetVersionFn>(symbol(o.handle, "cublasLtGetVersion"));
        std::size_t version = 0;
        if (c.source == "system" &&
            std::any_of(bundled.begin(), bundled.end(), [&](const Candidate& b) { return same_file(file, b.path); })) {
          reason = "resolved to the bundled copy " + file;
        } else if (get_version == nullptr) {
          reason = file + " has no cublasLtGetVersion";
        } else if (version = get_version(); !cublaslt_compatible(version, build)) {
          reason = file + " is cuBLASLt " + cublaslt_version_string(version) +
                   (version / 10000 != build / 10000
                        ? ", this build needs major version " + std::to_string(build / 10000)
                        : ", older than the " + cublaslt_version_string(build) + " this build was compiled against");
        } else {
          for (int i = 0; i < kCublasLtEntryCount && reason.empty(); ++i) {
            a.entry[static_cast<std::size_t>(i)] = symbol(o.handle, kSymbolNames[i]);
            if (a.entry[static_cast<std::size_t>(i)] == nullptr) reason = file + " has no " + kSymbolNames[i];
          }
        }
        if (reason.empty()) {
          a.status.ok = true;
          a.status.info = CublasLtInfo{file, c.source, version};
          return a;  // never unloaded
        }
        a.entry = {};
        close_library(o.handle);
      }
    }
    rejected += "\n  " + c.path + " (" + c.source + "): " + reason;
  }
  std::string not_found;
  for (std::size_t i = 0; i < missing.size(); ++i) not_found += (i == 0 ? "" : ", ") + missing[i];
  a.status.error = "no usable cuBLASLt " + std::to_string(build / 10000) + ".x >= " + cublaslt_version_string(build) +
                   " (" + what + ")";
  if (candidates.empty()) a.status.error += "; no candidates";
  if (!rejected.empty()) a.status.error += "; rejected:" + rejected;
  if (!missing.empty()) a.status.error += "\n  missing (" + std::to_string(missing.size()) + "): " + not_found;
  return a;
}

// What a search is configured by, for its error message.
std::string describe(CudaLibPolicy policy, const Environment& env) {
  const std::string file = env_value(env, "MOSSINFER_CUBLASLT", current_os());
  return file.empty() ? std::string("MOSSINFER_CUDA_LIBS=") + cuda_lib_policy_name(policy) : "MOSSINFER_CUBLASLT=" + file;
}

Attempt find(CudaLibPolicy policy, const Environment& env) {
  const LibOs os = current_os();
  const int major = static_cast<int>(cublaslt_build_version() / 10000);
  const std::string dir = own_module_dir();
  return try_candidates(cublaslt_candidates(policy, env, dir, os, major), cublaslt_bundled_candidates(dir, os, major),
                        describe(policy, env));
}

struct Loaded {
  CublasLtStatus status;
  std::array<void*, kCublasLtEntryCount> entry{};
  std::string unavailable;  // cublaslt_unavailable_message()
};

// Leaked on purpose: the shim may be called from static destructors, and the library is never unloaded.
const Loaded* loaded() noexcept {
  static const Loaded* const l = []() noexcept -> const Loaded* {
    Loaded* r = new (std::nothrow) Loaded;
    if (r == nullptr) return nullptr;
    try {
      const Environment env = process_environment();
      Attempt a = find(parse_cuda_lib_policy(env_value(env, "MOSSINFER_CUDA_LIBS", current_os())), env);
      r->status = std::move(a.status);
      r->entry = a.entry;
    } catch (const std::exception& e) {
      r->status = CublasLtStatus{false, {}, e.what()};
    } catch (...) {
      r->status = CublasLtStatus{false, {}, "cuBLASLt search failed"};
    }
    if (!r->status.ok) {
      try {
        r->unavailable = "CUBLAS_STATUS_NOT_INITIALIZED (cuBLASLt is not loaded: " + r->status.error + ")";
      } catch (...) {
      }
    }
    return r;
  }();
  return l;
}

}  // namespace

CudaLibPolicy parse_cuda_lib_policy(std::string_view value) {
  if (value.empty() || value == "auto") return CudaLibPolicy::kAuto;
  if (value == "system") return CudaLibPolicy::kSystem;
  if (value == "bundled") return CudaLibPolicy::kBundled;
  throw CudaLibraryError("MOSSINFER_CUDA_LIBS=" + std::string(value) + ": expected auto, system or bundled");
}

const char* cuda_lib_policy_name(CudaLibPolicy policy) {
  switch (policy) {
    case CudaLibPolicy::kSystem: return "system";
    case CudaLibPolicy::kBundled: return "bundled";
    case CudaLibPolicy::kAuto: break;
  }
  return "auto";
}

bool cublaslt_compatible(std::size_t runtime_version, std::size_t build_version) {
  return runtime_version / 10000 == build_version / 10000 && runtime_version >= build_version;
}

std::string cublaslt_version_string(std::size_t v) {
  return std::to_string(v / 10000) + "." + std::to_string(v / 100 % 100) + "." + std::to_string(v % 100);
}

std::size_t cublaslt_build_version() { return CUBLAS_VERSION; }

Environment process_environment() {
  Environment env;
#ifdef _WIN32
  wchar_t* block = GetEnvironmentStringsW();
  if (block == nullptr) return env;
  for (const wchar_t* p = block; *p != L'\0'; p += std::wcslen(p) + 1) {
    const std::wstring entry(p);
    const std::size_t eq = entry.find(L'=', 1);  // "=C:=C:\..." entries start with '='
    if (entry[0] == L'=' || eq == std::wstring::npos) continue;
    try {  // a variable that does not convert (unpaired surrogates) is skipped, never the whole search
      env.emplace_back(util::path_utf8(fs::path(entry.substr(0, eq))), util::path_utf8(fs::path(entry.substr(eq + 1))));
    } catch (const std::exception&) {
    }
  }
  FreeEnvironmentStringsW(block);
#else
  for (char** p = environ; p != nullptr && *p != nullptr; ++p) {
    const std::string_view entry(*p);
    const std::size_t eq = entry.find('=');
    if (eq == std::string_view::npos || eq == 0) continue;
    env.emplace_back(std::string(entry.substr(0, eq)), std::string(entry.substr(eq + 1)));
  }
#endif
  return env;
}

std::string env_value(const Environment& env, std::string_view name, LibOs os) {
  for (const auto& [k, v] : env) {
    if (k == name || (os == LibOs::kWindows && lower(k) == lower(std::string(name)))) return v;
  }
  return {};
}

std::vector<Candidate> cublaslt_bundled_candidates(const std::string& module_dir, LibOs os, int cuda_major) {
  std::vector<Candidate> out;
  if (module_dir.empty()) return out;
  const std::string lib = library_name(os, cuda_major);
  add(out, join(module_dir, lib, os), "bundled", os);
  // Executables in bin/ that link the core statically (moss-infer, tests): the package's lib/.
  if (os == LibOs::kLinux) add(out, join(join(join(module_dir, "..", os), "lib", os), lib, os), "bundled", os);
  return out;
}

std::vector<Candidate> cublaslt_candidates(CudaLibPolicy policy, const Environment& env, const std::string& module_dir,
                                           LibOs os, int cuda_major) {
  std::vector<Candidate> out;
  if (std::string file = env_value(env, "MOSSINFER_CUBLASLT", os); !file.empty()) {
    out.push_back(Candidate{std::move(file), "env"});
    return out;
  }
  // auto: the package's own copy first (reproducible results, certificates that apply); a package without one (the
  // system-CUDA variant) falls through to a compatible system copy.
  const std::vector<Candidate> bundled = cublaslt_bundled_candidates(module_dir, os, cuda_major);
  if (policy != CudaLibPolicy::kSystem) {
    for (const Candidate& c : bundled) add(out, c.path, "bundled", os);
  }
  if (policy != CudaLibPolicy::kBundled) system_candidates(out, env, os, cuda_major, bundled);
  return out;
}

LibOs current_os() {
#ifdef _WIN32
  return LibOs::kWindows;
#else
  return LibOs::kLinux;
#endif
}

std::string own_module_dir() {
  static const char anchor = 0;  // any address inside this module
#ifdef _WIN32
  HMODULE h = nullptr;
  if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                         reinterpret_cast<LPCWSTR>(&anchor), &h) == 0) {
    return {};
  }
  std::wstring buf(32768, L'\0');
  const DWORD n = GetModuleFileNameW(h, buf.data(), static_cast<DWORD>(buf.size()));
  if (n == 0 || n >= buf.size()) return {};
  buf.resize(n);
  return util::path_utf8(fs::path(buf).parent_path());
#else
  Dl_info info{};
  struct link_map* lm = nullptr;
  if (dladdr1(&anchor, &info, reinterpret_cast<void**>(&lm), RTLD_DL_LINKMAP) == 0) return {};
  // The main program's link map has no name (dladdr reports argv[0] for it): the kernel knows the file.
  std::string file;
  if (lm != nullptr && lm->l_name != nullptr && lm->l_name[0] != '\0') {
    file = lm->l_name;
  } else {
    std::error_code ec;
    file = fs::read_symlink("/proc/self/exe", ec).string();
    if (ec) return {};
  }
  std::error_code ec;
  const fs::path canonical = fs::canonical(file, ec);
  return (ec ? fs::path(file) : canonical).parent_path().string();
#endif
}

const CublasLtInfo& load_cublaslt() {
  const Loaded* l = loaded();
  if (l == nullptr) throw std::bad_alloc();
  if (!l->status.ok) throw CudaLibraryError(l->status.error);
  return l->status.info;
}

CublasLtStatus cublaslt_status() noexcept {
  try {
    const Loaded* l = loaded();
    return l != nullptr ? l->status : CublasLtStatus{false, {}, "out of memory"};
  } catch (...) {
    return CublasLtStatus{};
  }
}

CublasLtStatus find_cublaslt(CudaLibPolicy policy, const Environment& env) { return find(policy, env).status; }

CublasLtStatus try_cublaslt_candidates(const std::vector<Candidate>& candidates, const std::vector<Candidate>& bundled) {
  return try_candidates(candidates, bundled, "explicit candidates").status;
}

void* const* cublaslt_entry_points() noexcept {
  const Loaded* l = loaded();
  return l != nullptr && l->status.ok ? l->entry.data() : nullptr;
}

const char* cublaslt_unavailable_message() noexcept {
  const Loaded* l = loaded();
  if (l == nullptr) return "CUBLAS_STATUS_NOT_INITIALIZED (cuBLASLt is not loaded: out of memory)";
  return l->unavailable.c_str();
}

}  // namespace mi::backend
