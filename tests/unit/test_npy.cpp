#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>

#include "test_util.h"
#include "util/npy.h"

namespace {
const std::filesystem::path kGolden = std::filesystem::path(MOSSINFER_SOURCE_DIR) / "tests/golden";

// A version 1.0 .npy file: magic, u16 header length, header text, then `data`.
void write_npy(const std::filesystem::path& path, const std::string& header, const std::string& data) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out << "\x93NUMPY" << '\x01' << '\x00';
  out.put(static_cast<char>(header.size() & 0xFF));
  out.put(static_cast<char>(header.size() >> 8));
  out << header << data;
}
}  // namespace

TEST(Npy, LoadsGoldenNoise) {
  auto arr = mi::npy::load(kGolden / "noise/seed42_1024x1024.npy");
  EXPECT_EQ(arr.dtype, "f4");
  ASSERT_EQ(arr.shape, (std::vector<std::int64_t>{1, 128, 64, 64}));
  EXPECT_EQ(arr.numel(), 128 * 64 * 64);
  // torch.manual_seed(42); torch.randn(...) starts with 1.9269153.
  EXPECT_FLOAT_EQ(arr.to_vector<float>()[0], 1.9269153f);
}

TEST(Npy, LoadsGoldenTokenIds) {
  auto arr = mi::npy::load(kGolden / "g_short_1024.cond.token_ids.npy");
  EXPECT_EQ(arr.dtype, "i8");
  ASSERT_EQ(arr.shape.size(), 1u);
  EXPECT_EQ(arr.to_vector<std::int64_t>()[0], 248045);  // <|im_start|>
}

TEST(Npy, RejectsMissingFile) {
  EXPECT_THROW(mi::npy::load(kGolden / "does_not_exist.npy"), std::runtime_error);
}

TEST(Npy, ToVectorChecksDtype) {
  auto arr = mi::npy::load(kGolden / "g_short_1024.cond.token_ids.npy");
  EXPECT_THROW(arr.to_vector<double>(), std::runtime_error);         // f8, same size
  EXPECT_THROW(arr.to_vector<std::uint64_t>(), std::runtime_error);  // u8, same size
  EXPECT_THROW(arr.to_vector<std::int32_t>(), std::runtime_error);
  EXPECT_EQ(arr.to_vector<std::int64_t>().size(), static_cast<std::size_t>(arr.numel()));
}

TEST(Npy, RejectsMalformedHeadersAsRuntimeError) {
  mi::testing::TempDir dir("npy_bad");
  const auto path = dir.path() / "a.npy";
  const std::string ok_header = "{'descr': '<f4', 'fortran_order': False, 'shape': (2,), }";
  write_npy(path, ok_header, std::string(8, '\0'));
  EXPECT_EQ(mi::npy::load(path).to_vector<float>(), (std::vector<float>{0.0f, 0.0f}));

  auto bad = [&](const std::string& header, const std::string& data, const char* what) {
    write_npy(path, header, data);
    EXPECT_THROW(mi::npy::load(path), std::runtime_error) << what;
  };
  bad("{'descr': '<f4', 'fortran_order': False, 'shape': (x,), }", std::string(8, '\0'), "non-numeric dim");
  bad("{'descr': '<f4', 'fortran_order': False, 'shape': (-2,), }", std::string(8, '\0'), "negative dim");
  bad("{'descr': '<f4', 'fortran_order': False, 'shape': (99999999999999999999,), }", "", "dim out of int64 range");
  bad("{'descr': '<f4', 'fortran_order': False, 'shape': (4611686018427387904, 4), }", "", "numel overflow");
  bad("{'descr': '<f4', 'fortran_order': False, 'shape': (2305843009213693952,), }", "", "numel x itemsize overflow");
  bad("{'descr': '<f4', 'fortran_order': False, 'shape': (1000000,), }", std::string(8, '\0'), "data shorter than shape");
  bad("{'descr': '<fX', 'fortran_order': False, 'shape': (2,), }", std::string(8, '\0'), "non-numeric itemsize");
  bad("{'descr': '<f3', 'fortran_order': False, 'shape': (2,), }", std::string(8, '\0'), "odd itemsize");

  // Header length field cut off, and a header length past the end of the file.
  std::ofstream(path, std::ios::binary | std::ios::trunc) << "\x93NUMPY" << '\x01' << '\x00' << 'x';
  EXPECT_THROW(mi::npy::load(path), std::runtime_error);
  std::ofstream(path, std::ios::binary | std::ios::trunc) << "\x93NUMPY" << '\x02' << '\x00' << "\xff\xff\xff\x7f{}";
  EXPECT_THROW(mi::npy::load(path), std::runtime_error);
}
