// Where quality certificates are searched (pipeline/quality_cert.h certificate_files): the caller's directories, the
// cache directory, the DiT's sidecar and the install tree resolved from the engine module, with the harness fixture
// (tests/data/quality_cert) as the certificate shipped in an install tree. CPU only.
#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "backend/cuda_libs.h"
#include "pipeline/quality_cert.h"
#include "test_util.h"

namespace {

namespace fs = std::filesystem;
using namespace mi::pipeline;

const fs::path kData = fs::path(MOSSINFER_SOURCE_DIR) / "tests/data/quality_cert";
constexpr std::uint64_t kFixtureCublasLt = 130800;  // the harness fixture's "cublaslt" numerics entry

void write_bytes(const fs::path& p, const std::string& s) {
  std::ofstream out(p, std::ios::binary | std::ios::trunc);
  out << s;
}

// Precedence: the caller's directories in order, <cache_dir>/quality_certs, the sidecar, then the install tree; a
// directory reached twice is listed once.
TEST(QualityCertLocations, Precedence) {
  mi::testing::TempDir dir("qcert_order");
  const fs::path dit = dir.path() / "dit.safetensors";
  write_bytes(dit, "x");
  const fs::path cache = dir.path() / "cache";
  const fs::path install = dir.path() / "prefix" / "share" / "mossinfer" / "quality_certs";
  const fs::path extra1 = dir.path() / "extra1", extra2 = dir.path() / "extra2";
  for (const fs::path& d : {cache / "quality_certs", install, extra1, extra2}) fs::create_directories(d);
  write_bytes(cache / "quality_certs" / "c.json", "{}");
  write_bytes(install / "i.json", "{}");
  write_bytes(extra1 / "e1.json", "{}");
  write_bytes(extra2 / "e2.json", "{}");
  write_bytes(certificate_sidecar(dit), "{}");
  const std::vector<fs::path> files = certificate_files(dit, cache, {extra2, extra1, cache / "quality_certs"}, install);
  ASSERT_EQ(files.size(), 5u);
  EXPECT_EQ(files[0], extra2 / "e2.json");
  EXPECT_EQ(files[1], extra1 / "e1.json");
  EXPECT_EQ(files[2], cache / "quality_certs" / "c.json");
  EXPECT_EQ(files[3], certificate_sidecar(dit));
  EXPECT_EQ(files[4], install / "i.json");
  EXPECT_EQ(certificate_files(dit, {}, {}, {}).size(), 1u) << "no install dir: the sidecar only";
}

// The install tree's directory sits next to the engine module's lib/, lib64/ or bin/ directory of an install prefix and
// is resolved from the module at run time; a module anywhere else has no install tree to search (and says so).
TEST(QualityCertLocations, InstallTreeDirectory) {
  EXPECT_EQ(install_certificate_dir(fs::path("/opt/mossinfer/lib")), fs::path("/opt/mossinfer/share/mossinfer/quality_certs"));
  EXPECT_EQ(install_certificate_dir(fs::path("/opt/mossinfer/lib64")), fs::path("/opt/mossinfer/share/mossinfer/quality_certs"));
  EXPECT_EQ(install_certificate_dir(fs::path("/opt/mossinfer/bin/")), fs::path("/opt/mossinfer/share/mossinfer/quality_certs"));
  EXPECT_TRUE(install_certificate_dir(fs::path("/opt/app")).empty()) << "not /opt/share";
  EXPECT_TRUE(install_certificate_dir(fs::path()).empty());
  const std::string module = mi::backend::own_module_dir();
  ASSERT_FALSE(module.empty());
  const fs::path dir = install_certificate_dir();
  EXPECT_EQ(dir, install_certificate_dir(mi::util::path_from_utf8(module)));
  EXPECT_EQ(install_certificate_note().empty(), !dir.empty()) << install_certificate_note();
}

// The harness fixture's certificate shipped in an install tree is found and matched; a mismatched one beside it is
// rejected with its reason, and the same certificate in a caller's directory takes precedence.
TEST(QualityCertLocations, FindsTheInstallTreeCertificate) {
  mi::testing::TempDir dir("qcert_install");
  const fs::path dit = dir.path() / "renamed_dit.safetensors";  // renaming keeps the identity
  fs::copy_file(kData / "tiny.safetensors", dit);
  const std::vector<Certificate> fixture = read_certificate_file(certificate_sidecar(kData / "tiny.safetensors"));
  ASSERT_EQ(fixture.size(), 1u);
  const Certificate& c = fixture[0];
  const CertificateQuery q{c.config, c.gpu_arch, engine_numerics(c.config, kFixtureCublasLt), std::nullopt};
  const fs::path install = dir.path() / "prefix" / "share" / "mossinfer" / "quality_certs";
  fs::create_directories(install);

  Certificate other = c;
  other.gpu_arch = "sm_86";
  ASSERT_TRUE(write_certificate_file(install / "a_other_gpu.json", {other}));
  CertificateSearch r = find_certificate(dit, {}, q, {}, nullptr, {}, install);
  EXPECT_FALSE(r.certificate.has_value());
  EXPECT_NE(r.reason.find("a_other_gpu.json"), std::string::npos) << r.reason;
  EXPECT_NE(r.reason.find("sm_86"), std::string::npos) << r.reason;
  EXPECT_FALSE(find_certificate(dit, {}, q, {}).certificate.has_value()) << "this build's install tree holds none";

  fs::copy_file(certificate_sidecar(kData / "tiny.safetensors"), install / "b_fixture.json");
  std::optional<ModelIdentity> full;
  r = find_certificate(dit, {}, q, {}, &full, {}, install);
  ASSERT_TRUE(r.certificate.has_value()) << r.reason;
  EXPECT_EQ(r.file, install / "b_fixture.json");
  EXPECT_EQ(r.certificate->gpu_arch, c.gpu_arch);
  ASSERT_TRUE(full.has_value());
  EXPECT_TRUE(full->dit.same_content(c.model.dit));
  CertificateQuery elsewhere = q;
  elsewhere.gpu_arch = "sm_120";
  r = find_certificate(dit, {}, elsewhere, {}, nullptr, {}, install);
  EXPECT_FALSE(r.certificate.has_value());
  EXPECT_NE(r.reason.find("b_fixture.json"), std::string::npos) << r.reason;
  EXPECT_NE(r.reason.find("sm_120"), std::string::npos) << r.reason;

  const fs::path extra = dir.path() / "extra";
  fs::create_directories(extra);
  fs::copy_file(install / "b_fixture.json", extra / "x.json");
  r = find_certificate(dit, {}, q, {}, nullptr, {extra}, install);
  ASSERT_TRUE(r.certificate.has_value()) << r.reason;
  EXPECT_EQ(r.file, extra / "x.json");
}

}  // namespace
