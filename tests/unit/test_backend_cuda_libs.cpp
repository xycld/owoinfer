// src/backend/cuda_libs: the cuBLASLt search (policy, candidate lists, compatibility rule) on injected inputs, and real
// searches on this machine. CPU only: loading cuBLASLt creates no CUDA context (the cpu guard checks it).
#include <gtest/gtest.h>

#include <cublasLt.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#ifndef _WIN32
#include <dlfcn.h>
#endif

#include "backend/cuda_libs.h"
#include "test_util.h"
#include "util/path.h"

namespace {

namespace fs = std::filesystem;
using mi::backend::Candidate;
using mi::backend::CudaLibPolicy;
using mi::backend::Environment;
using mi::backend::LibOs;

std::vector<std::string> paths(const std::vector<Candidate>& c, const char* source = nullptr) {
  std::vector<std::string> out;
  for (const Candidate& x : c) {
    if (source == nullptr || x.source == source) out.push_back(x.path);
  }
  return out;
}

// The process environment without the variables that steer the search.
Environment clean_environment() {
  Environment env;
  for (auto& kv : mi::backend::process_environment()) {
    if (kv.first != "MOSSINFER_CUBLASLT" && kv.first != "MOSSINFER_CUDA_LIBS") env.push_back(kv);
  }
  return env;
}

TEST(CudaLibs, ParsePolicy) {
  EXPECT_EQ(mi::backend::parse_cuda_lib_policy(""), CudaLibPolicy::kAuto);
  EXPECT_EQ(mi::backend::parse_cuda_lib_policy("auto"), CudaLibPolicy::kAuto);
  EXPECT_EQ(mi::backend::parse_cuda_lib_policy("system"), CudaLibPolicy::kSystem);
  EXPECT_EQ(mi::backend::parse_cuda_lib_policy("bundled"), CudaLibPolicy::kBundled);
  for (const char* bad : {"System", "both", " auto", "1"}) {
    EXPECT_THROW(mi::backend::parse_cuda_lib_policy(bad), mi::backend::CudaLibraryError) << bad;
  }
  try {
    mi::backend::parse_cuda_lib_policy("both");
  } catch (const mi::backend::CudaLibraryError& e) {
    EXPECT_NE(std::string(e.what()).find("MOSSINFER_CUDA_LIBS=both"), std::string::npos) << e.what();
  }
}

TEST(CudaLibs, CompatibilityRule) {
  EXPECT_TRUE(mi::backend::cublaslt_compatible(130800, 130800));
  EXPECT_TRUE(mi::backend::cublaslt_compatible(130801, 130800));  // newer patch
  EXPECT_TRUE(mi::backend::cublaslt_compatible(131000, 130800));  // newer minor
  EXPECT_FALSE(mi::backend::cublaslt_compatible(130700, 130800));  // older minor
  EXPECT_FALSE(mi::backend::cublaslt_compatible(130100, 130204));
  EXPECT_FALSE(mi::backend::cublaslt_compatible(120900, 130800));  // other major, older
  EXPECT_FALSE(mi::backend::cublaslt_compatible(140000, 130800));  // other major, newer
  EXPECT_FALSE(mi::backend::cublaslt_compatible(0, 130800));
  EXPECT_EQ(mi::backend::cublaslt_version_string(130800), "13.8.0");
  EXPECT_EQ(mi::backend::cublaslt_version_string(120804), "12.8.4");
  EXPECT_EQ(mi::backend::cublaslt_build_version(), static_cast<std::size_t>(CUBLAS_VERSION));
}

TEST(CudaLibs, LinuxCandidates) {
  // CUDA_HOME repeats CUDA_PATH (trailing slash) and /opt/cuda: each file is tried once. Names are case-sensitive.
  const Environment env{{"CUDA_PATH", "/opt/cuda/"}, {"CUDA_HOME", "/opt/cuda"}, {"cuda_path", "/ignored"}};
  const std::string lib = "libcublasLt.so.13";
  const std::vector<std::string> system{lib,
                                        "/opt/cuda/lib64/" + lib,
                                        "/opt/cuda/targets/x86_64-linux/lib/" + lib,
                                        "/usr/local/cuda-13/lib64/" + lib,
                                        "/usr/local/cuda-13/targets/x86_64-linux/lib/" + lib,
                                        "/usr/local/cuda/lib64/" + lib,
                                        "/usr/local/cuda/targets/x86_64-linux/lib/" + lib,
                                        "/usr/lib/x86_64-linux-gnu/" + lib,  // distribution packages, by path
                                        "/usr/lib64/" + lib,
                                        "/usr/lib/" + lib};
  const std::vector<std::string> bundled{"/pkg/lib/" + lib};  // "/pkg/lib/../lib" is the same directory
#if defined(__x86_64__) || defined(_M_X64)
  const auto aut = mi::backend::cublaslt_candidates(CudaLibPolicy::kAuto, env, "/pkg/lib", LibOs::kLinux, 13);
  std::vector<std::string> both = bundled;  // auto: the package's own copy first, then the system locations
  both.insert(both.end(), system.begin(), system.end());
  EXPECT_EQ(paths(aut), both);
  EXPECT_EQ(paths(aut, "system"), system);
  EXPECT_EQ(paths(aut, "bundled"), bundled);
  EXPECT_EQ(paths(mi::backend::cublaslt_candidates(CudaLibPolicy::kSystem, env, "/pkg/lib", LibOs::kLinux, 13)), system);
#endif
  EXPECT_EQ(paths(mi::backend::cublaslt_candidates(CudaLibPolicy::kBundled, env, "/pkg/lib", LibOs::kLinux, 13)), bundled);
  // An executable in bin/ that links the core statically looks in the package's lib/ too.
  EXPECT_EQ(paths(mi::backend::cublaslt_candidates(CudaLibPolicy::kBundled, {}, "/pkg/bin", LibOs::kLinux, 12)),
            (std::vector<std::string>{"/pkg/bin/libcublasLt.so.12", "/pkg/bin/../lib/libcublasLt.so.12"}));
  // No environment: the fixed locations only. Unknown module directory: no bundled candidates.
  const auto plain = mi::backend::cublaslt_candidates(CudaLibPolicy::kAuto, {}, "", LibOs::kLinux, 13);
  EXPECT_EQ(plain.front(), (Candidate{lib, "system"}));
  EXPECT_EQ(plain.back(), (Candidate{"/usr/lib/" + lib, "system"}));
  EXPECT_TRUE(paths(plain, "bundled").empty());
  // Relative CUDA_PATH / CUDA_HOME roots are never searched (they would depend on the working directory).
  const auto relative = mi::backend::cublaslt_candidates(CudaLibPolicy::kSystem, {{"CUDA_PATH", "cuda"}, {"CUDA_HOME", "./c"}},
                                                         "", LibOs::kLinux, 13);
  EXPECT_EQ(paths(relative), paths(mi::backend::cublaslt_candidates(CudaLibPolicy::kSystem, {}, "", LibOs::kLinux, 13)));
  // A system location that is the bundled file (a package unpacked into /usr/lib) is the bundled copy: the system
  // entry goes, the bundled one stays, under every policy.
  for (const CudaLibPolicy p : {CudaLibPolicy::kAuto, CudaLibPolicy::kSystem}) {
    const auto in_usr = mi::backend::cublaslt_candidates(p, {}, "/usr/lib", LibOs::kLinux, 13);
    EXPECT_EQ(std::count(in_usr.begin(), in_usr.end(), Candidate{"/usr/lib/" + lib, "system"}), 0);
    EXPECT_EQ(std::count(in_usr.begin(), in_usr.end(), Candidate{"/usr/lib/" + lib, "bundled"}),
              p == CudaLibPolicy::kAuto ? 1 : 0);
  }
  // MOSSINFER_CUBLASLT is the only candidate, whatever the policy.
  Environment with_file = env;
  with_file.emplace_back("MOSSINFER_CUBLASLT", "/x/libcublasLt.so.13.9");
  for (const CudaLibPolicy p : {CudaLibPolicy::kAuto, CudaLibPolicy::kSystem, CudaLibPolicy::kBundled}) {
    EXPECT_EQ(mi::backend::cublaslt_candidates(p, with_file, "/pkg/lib", LibOs::kLinux, 13),
              (std::vector<Candidate>{{"/x/libcublasLt.so.13.9", "env"}}));
  }
  // Set but empty is unset.
  with_file.back().second = "";
  EXPECT_EQ(paths(mi::backend::cublaslt_candidates(CudaLibPolicy::kBundled, with_file, "/pkg/lib", LibOs::kLinux, 13)),
            bundled);
}

TEST(CudaLibs, WindowsCandidates) {
  // Names are case-insensitive; CUDA_PATH_V13_* newest minor first; other majors and malformed names are ignored.
  const Environment env{{"Cuda_Path", "C:\\CUDA\\v13.3\\"},
                        {"CUDA_PATH_V13_0", "C:\\CUDA\\v13.0"},
                        {"CUDA_PATH_V13_4", "relative\\cuda"},  // not absolute: skipped
                        {"CUDA_PATH_V13_3", "C:\\CUDA\\v13.3"},
                        {"cuda_path_v13_10", "D:/cuda/v13.10"},
                        {"CUDA_PATH_V12_8", "C:\\CUDA\\v12.8"},
                        {"CUDA_PATH_V13_X", "C:\\bad"},
                        {"CUDA_PATH_V13_2", ""}};
  const std::vector<std::string> system{
      "C:\\CUDA\\v13.3\\bin\\x64\\cublasLt64_13.dll", "C:\\CUDA\\v13.3\\bin\\cublasLt64_13.dll",
      "D:/cuda/v13.10\\bin\\x64\\cublasLt64_13.dll",  "D:/cuda/v13.10\\bin\\cublasLt64_13.dll",
      "C:\\CUDA\\v13.0\\bin\\x64\\cublasLt64_13.dll", "C:\\CUDA\\v13.0\\bin\\cublasLt64_13.dll"};
  const std::vector<std::string> bundled{"C:\\Program Files\\App\\cublasLt64_13.dll"};
  const std::string dir = "C:\\Program Files\\App";
  const auto aut = mi::backend::cublaslt_candidates(CudaLibPolicy::kAuto, env, dir, LibOs::kWindows, 13);
  EXPECT_EQ(paths(aut, "system"), system);
  EXPECT_EQ(paths(aut, "bundled"), bundled);
  EXPECT_EQ(aut.front().source, "bundled");  // auto: bundled first
  EXPECT_EQ(aut.back().source, "system");
  EXPECT_EQ(paths(mi::backend::cublaslt_candidates(CudaLibPolicy::kSystem, env, dir, LibOs::kWindows, 13)), system);
  EXPECT_EQ(paths(mi::backend::cublaslt_candidates(CudaLibPolicy::kBundled, env, dir, LibOs::kWindows, 13)), bundled);
  // Never a bare name on Windows (the application directory would come first): every candidate has a directory.
  EXPECT_TRUE(mi::backend::cublaslt_candidates(CudaLibPolicy::kSystem, {}, dir, LibOs::kWindows, 13).empty());
  Environment with_file = env;
  with_file.emplace_back("mossinfer_cublaslt", "E:\\lt\\cublasLt64_13.dll");
  EXPECT_EQ(mi::backend::cublaslt_candidates(CudaLibPolicy::kAuto, with_file, dir, LibOs::kWindows, 13),
            (std::vector<Candidate>{{"E:\\lt\\cublasLt64_13.dll", "env"}}));
}

// This platform's library file name, and an absolute path to it in a directory that does not exist.
std::string lib_name() {
  return mi::backend::current_os() == LibOs::kWindows ? "cublasLt64_" + std::to_string(CUBLAS_VER_MAJOR) + ".dll"
                                                      : "libcublasLt.so." + std::to_string(CUBLAS_VER_MAJOR);
}
std::string missing_file() {
  return (mi::backend::current_os() == LibOs::kWindows ? "C:\\nonexistent\\dir\\" : "/nonexistent/dir/") + lib_name();
}

TEST(CudaLibs, EnvOverrideToAMissingFileListsWhyItFailed) {
  const std::string file = missing_file();
  Environment env = clean_environment();
  env.emplace_back("MOSSINFER_CUBLASLT", file);
  const mi::backend::CublasLtStatus s = mi::backend::find_cublaslt(CudaLibPolicy::kSystem, env);
  EXPECT_FALSE(s.ok);
  EXPECT_NE(s.error.find("MOSSINFER_CUBLASLT=" + file), std::string::npos) << s.error;
  EXPECT_NE(s.error.find("missing (1): " + file + " (env)"), std::string::npos) << s.error;
  EXPECT_EQ(s.error.find("(system)"), std::string::npos) << "the override is the only candidate: " << s.error;
  // A relative override is refused: it names one file, never a search.
  Environment relative = clean_environment();
  relative.emplace_back("MOSSINFER_CUBLASLT", lib_name());
  const mi::backend::CublasLtStatus r = mi::backend::find_cublaslt(CudaLibPolicy::kAuto, relative);
  EXPECT_FALSE(r.ok);
  EXPECT_NE(r.error.find(lib_name() + " (env): not an absolute path"), std::string::npos) << r.error;
}

TEST(CudaLibs, BundledSearchFromTheBuildTree) {
  // The test binary has no cuBLASLt next to it: `bundled` lists both locations it tried.
  const std::string dir = mi::backend::own_module_dir();
  ASSERT_FALSE(dir.empty());
  EXPECT_TRUE(fs::is_directory(dir)) << dir;
  const mi::backend::CublasLtStatus s = mi::backend::find_cublaslt(CudaLibPolicy::kBundled, clean_environment());
  if (s.ok) {
    EXPECT_EQ(s.info.source, "bundled");
    GTEST_SKIP() << "a cuBLASLt copy sits next to the test binary: " << s.info.path;
  }
  EXPECT_NE(s.error.find("MOSSINFER_CUDA_LIBS=bundled"), std::string::npos) << s.error;
  EXPECT_NE(s.error.find("(bundled)"), std::string::npos) << s.error;
  EXPECT_NE(s.error.find("missing ("), std::string::npos) << s.error;
  EXPECT_EQ(s.error.find("(system)"), std::string::npos) << s.error;
}

TEST(CudaLibs, NotACublasLtLibraryIsRejected) {
  // A file that is no library at all.
  mi::testing::TempDir dir("cuda_libs_not_a_library");
  const fs::path text = dir.path() / lib_name();
  std::ofstream(text) << "not a library\n";
  Environment env = clean_environment();
  env.emplace_back("MOSSINFER_CUBLASLT", mi::util::path_utf8(fs::absolute(text)));
  mi::backend::CublasLtStatus s = mi::backend::find_cublaslt(CudaLibPolicy::kAuto, env);
  EXPECT_FALSE(s.ok);
  EXPECT_NE(s.error.find("(env): not loadable"), std::string::npos) << s.error;
#ifndef _WIN32
  // A real library without cuBLASLt's entry points: the C library this process uses.
  Dl_info info{};
  ASSERT_NE(dladdr(reinterpret_cast<void*>(&::printf), &info), 0);
  ASSERT_NE(info.dli_fname, nullptr);
  env.back().second = fs::canonical(info.dli_fname).string();
  s = mi::backend::find_cublaslt(CudaLibPolicy::kAuto, env);
  EXPECT_FALSE(s.ok);
  EXPECT_NE(s.error.find("has no cublasLtGetVersion"), std::string::npos) << s.error;
#endif
}

// The loader's branches on stand-in libraries (tests/unit/test_backend_fake_cublaslt.c): each rejection is reported
// with its reason, the rejected library is closed again, and the search goes on to the next candidate.
#if defined(MI_FAKE_CUBLASLT_GOOD) && !defined(_WIN32)
bool still_loaded(const char* file) {
  void* h = dlopen(file, RTLD_NOW | RTLD_NOLOAD);
  if (h != nullptr) dlclose(h);  // drop the reference RTLD_NOLOAD took
  return h != nullptr;
}

TEST(CudaLibs, FakeLibrariesDriveEveryBranch) {
  const std::size_t build = CUBLAS_VERSION;
  const std::string v = mi::backend::cublaslt_version_string(build);
  const auto status = [](std::vector<Candidate> c, std::vector<Candidate> bundled = {}) {
    return mi::backend::try_cublaslt_candidates(c, bundled);
  };
  // Every rejection, in order, then the "missing" summary; nothing stays loaded.
  const std::vector<Candidate> bad{{MI_FAKE_CUBLASLT_MAJOR, "system"}, {MI_FAKE_CUBLASLT_NOSYM, "system"},
                                   {"/nonexistent/libcublasLt.so.13", "system"}, {MI_FAKE_CUBLASLT_OLD, "system"}};
  const mi::backend::CublasLtStatus r = status(bad);
  ASSERT_FALSE(r.ok);
  const std::size_t major_at = r.error.find("is cuBLASLt " + mi::backend::cublaslt_version_string(build + 10000) +
                                            ", this build needs major version " + std::to_string(build / 10000));
  const std::size_t nosym_at = r.error.find("has no cublasLtMatmulAlgoConfigGetAttribute");
  const std::size_t missing_at = r.error.find("missing (1): /nonexistent/libcublasLt.so.13 (system)");
  EXPECT_NE(major_at, std::string::npos) << r.error;
  EXPECT_NE(nosym_at, std::string::npos) << r.error;
  EXPECT_NE(missing_at, std::string::npos) << r.error;
  EXPECT_LT(major_at, nosym_at) << r.error;
  EXPECT_LT(nosym_at, missing_at) << "real rejections first, the missing files last: " << r.error;
  if (build % 10000 != 0) {
    const std::size_t old_at = r.error.find(", older than the " + v + " this build was compiled against");
    EXPECT_NE(old_at, std::string::npos) << r.error;
    EXPECT_LT(old_at, missing_at) << r.error;
  }
  for (const char* f : {MI_FAKE_CUBLASLT_MAJOR, MI_FAKE_CUBLASLT_NOSYM, MI_FAKE_CUBLASLT_OLD}) {
    EXPECT_FALSE(still_loaded(f)) << f << " was rejected but stays loaded";
  }

  // Close and continue: the usable copy after the rejected ones is taken.
  std::vector<Candidate> then_good = bad;
  then_good.push_back({MI_FAKE_CUBLASLT_GOOD, "system"});
  const mi::backend::CublasLtStatus g = status(then_good);
  ASSERT_TRUE(g.ok) << g.error;
  EXPECT_EQ(g.info.source, "system");
  EXPECT_EQ(g.info.version, build);
  EXPECT_EQ(fs::path(g.info.path), fs::canonical(MI_FAKE_CUBLASLT_GOOD));
  EXPECT_TRUE(still_loaded(MI_FAKE_CUBLASLT_GOOD)) << "the copy in use is never unloaded";
  for (const char* f : {MI_FAKE_CUBLASLT_MAJOR, MI_FAKE_CUBLASLT_NOSYM, MI_FAKE_CUBLASLT_OLD}) {
    EXPECT_FALSE(still_loaded(f)) << f;
  }

  // A "system" candidate that is the bundled file is rejected; the same file as the bundled candidate is taken.
  const std::vector<Candidate> bundled{{MI_FAKE_CUBLASLT_GOOD, "bundled"}};
  const mi::backend::CublasLtStatus only_system = status({{MI_FAKE_CUBLASLT_GOOD, "system"}}, bundled);
  EXPECT_FALSE(only_system.ok);
  EXPECT_NE(only_system.error.find("resolved to the bundled copy"), std::string::npos) << only_system.error;
  const mi::backend::CublasLtStatus both = status({{MI_FAKE_CUBLASLT_GOOD, "system"}, {MI_FAKE_CUBLASLT_GOOD, "bundled"}},
                                                  bundled);
  ASSERT_TRUE(both.ok) << both.error;
  EXPECT_EQ(both.info.source, "bundled");

  // The same through MOSSINFER_CUBLASLT.
  Environment env = clean_environment();
  env.emplace_back("MOSSINFER_CUBLASLT", MI_FAKE_CUBLASLT_NOSYM);
  const mi::backend::CublasLtStatus e = mi::backend::find_cublaslt(CudaLibPolicy::kAuto, env);
  EXPECT_FALSE(e.ok);
  EXPECT_NE(e.error.find("(env): "), std::string::npos) << e.error;
  EXPECT_NE(e.error.find("has no cublasLtMatmulAlgoConfigGetAttribute"), std::string::npos) << e.error;
  env.back().second = MI_FAKE_CUBLASLT_GOOD;
  const mi::backend::CublasLtStatus eg = mi::backend::find_cublaslt(CudaLibPolicy::kAuto, env);
  ASSERT_TRUE(eg.ok) << eg.error;
  EXPECT_EQ(eg.info.source, "env");
}
#endif

TEST(CudaLibs, SystemSearchOnThisMachine) {
  const mi::backend::CublasLtStatus s = mi::backend::find_cublaslt(CudaLibPolicy::kSystem, clean_environment());
  if (!s.ok) {
    EXPECT_NE(s.error.find("(system)"), std::string::npos) << s.error;
    GTEST_SKIP() << "no system cuBLASLt on this machine: " << s.error;
  }
  std::printf("[cuda_libs] system: %s %zu\n", s.info.path.c_str(), s.info.version);
  EXPECT_EQ(s.info.source, "system");
  EXPECT_TRUE(fs::exists(s.info.path)) << s.info.path;
  EXPECT_TRUE(mi::backend::cublaslt_compatible(s.info.version, CUBLAS_VERSION)) << s.info.version;
  // The dev box: CUDA in /opt/cuda, in the ld.so cache, and nothing on LD_LIBRARY_PATH.
  const fs::path dev = "/opt/cuda/lib64/libcublasLt.so." + std::to_string(CUBLAS_VER_MAJOR);
  const bool ld_path = !mi::backend::env_value(mi::backend::process_environment(), "LD_LIBRARY_PATH", LibOs::kLinux).empty();
  if (mi::backend::current_os() == LibOs::kLinux && fs::exists(dev) && !ld_path) {
    EXPECT_EQ(fs::path(s.info.path), fs::canonical(dev));
  }
}

TEST(CudaLibs, ShimForwardsToTheProcessCopy) {
  const mi::backend::CublasLtStatus s = mi::backend::cublaslt_status();
  if (s.ok) {
    EXPECT_EQ(cublasLtGetVersion(), s.info.version);
    EXPECT_NE(mi::backend::cublaslt_entry_points(), nullptr);
    EXPECT_STREQ(mi::backend::cublaslt_unavailable_message(), "");
    EXPECT_EQ(std::string(cublasLtGetStatusString(CUBLAS_STATUS_SUCCESS)).find("not loaded"), std::string::npos);  // the library's own text
    EXPECT_EQ(&mi::backend::load_cublaslt(), &mi::backend::load_cublaslt());  // loaded once
    EXPECT_EQ(mi::backend::load_cublaslt().path, s.info.path);
  } else {
    // No usable copy: the C functions report it instead of throwing.
    EXPECT_EQ(cublasLtGetVersion(), 0u);
    EXPECT_EQ(mi::backend::cublaslt_entry_points(), nullptr);
    cublasLtHandle_t h = nullptr;
    EXPECT_EQ(cublasLtCreate(&h), CUBLAS_STATUS_NOT_INITIALIZED);
    EXPECT_NE(std::string(cublasLtGetStatusString(CUBLAS_STATUS_SUCCESS)).find("not loaded"), std::string::npos);
    EXPECT_THROW(mi::backend::load_cublaslt(), mi::backend::CudaLibraryError);
  }
}

}  // namespace
