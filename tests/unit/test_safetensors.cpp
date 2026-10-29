#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <thread>

#ifdef __linux__
#include <fcntl.h>
#include <sys/vfs.h>
#include <sys/wait.h>
#include <unistd.h>

#include <csignal>
#endif
#ifndef _WIN32
#include <sys/stat.h>

#include <cerrno>
#else
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <winioctl.h>
#endif

#include "test_util.h"
#include "util/regular_file.h"
#include "util/replaceable_file.h"
#include "util/xxh64.h"
#include "weights/file_reader.h"
#include "weights/safetensors.h"
#include "weights/test_seams.h"
#include "weights/upload.h"

namespace {

using namespace mi::weights;
using mi::testing::TempDir;
using mi::testing::pattern_byte;
using mi::testing::write_raw;
using mi::testing::write_safetensors;

std::string xxh_hex(const std::string& s) {
  char b[17];
  std::snprintf(b, sizeof b, "%016llx", static_cast<unsigned long long>(mi::xxh64(s.data(), s.size())));
  return b;
}

// ---- XXH64 ------------------------------------------------------------------------------------

TEST(Xxh64, KnownVectors) {
  // Reference values from the xxhsum CLI (xxHash 0.8).
  EXPECT_EQ(xxh_hex(""), "ef46db3751d8e999");
  EXPECT_EQ(xxh_hex("a"), "d24ec4f1a98c6e5b");
  EXPECT_EQ(xxh_hex("abc"), "44bc2cf5ad770999");
  EXPECT_EQ(xxh_hex("Nobody inspects the spammish repetition"), "fbcea83c8a378bf1");
}

TEST(Xxh64, LengthsAroundStripeBoundary) {
  std::string d(1000, '\0');
  for (std::size_t i = 0; i < d.size(); ++i) d[i] = static_cast<char>(pattern_byte(i));
  EXPECT_EQ(xxh_hex(d.substr(0, 31)), "cec8c2be9bd26a69");
  EXPECT_EQ(xxh_hex(d.substr(0, 32)), "b69064153d4851a5");
  EXPECT_EQ(xxh_hex(d.substr(0, 33)), "732da22315773994");
  EXPECT_EQ(xxh_hex(d.substr(0, 100)), "1c0c2332e8ff479a");
  EXPECT_EQ(xxh_hex(d), "c7227a16ad9064c5");
}

TEST(Xxh64, StreamingMatchesOneShotForAnyChunking) {
  std::string d(1000, '\0');
  for (std::size_t i = 0; i < d.size(); ++i) d[i] = static_cast<char>(pattern_byte(i));
  const std::uint64_t want = mi::xxh64(d.data(), d.size());
  for (std::size_t chunk : {1u, 7u, 31u, 32u, 33u, 64u, 999u}) {
    mi::Xxh64 h;
    for (std::size_t pos = 0; pos < d.size(); pos += chunk) h.update(d.data() + pos, std::min(chunk, d.size() - pos));
    EXPECT_EQ(h.digest(), want) << "chunk " << chunk;
  }
}

// ---- safetensors parsing ----------------------------------------------------------------------

TEST(Safetensors, ParsesValidFile) {
  TempDir dir("st_valid");
  auto path = dir.path() / "a.safetensors";
  write_safetensors(path,
                    {{"w", DType::kBF16, {2, 3}}, {"scalar", DType::kF32, {}}, {"empty", DType::kU8, {0, 5}}, {"i", DType::kI64, {4}}},
                    "{\"format\":\"pt\"}");
  SafetensorsFile f(path);
  EXPECT_EQ(f.metadata().at("format"), "pt");
  ASSERT_EQ(f.tensors().size(), 4u);
  const TensorInfo* w = f.find("w");
  ASSERT_NE(w, nullptr);
  EXPECT_EQ(w->dtype, DType::kBF16);
  EXPECT_EQ(w->shape, (std::vector<std::int64_t>{2, 3}));
  EXPECT_EQ(w->nbytes, 12u);
  EXPECT_EQ(w->offset, f.data_start());
  EXPECT_EQ(f.find("scalar")->numel(), 1);
  EXPECT_EQ(f.find("scalar")->nbytes, 4u);
  EXPECT_EQ(f.find("empty")->nbytes, 0u);
  EXPECT_EQ(f.find("i")->offset, f.data_start() + 12 + 4);
  EXPECT_EQ(f.find("nope"), nullptr);

  // Offsets are absolute: reading through them yields the pattern at the data-relative position.
  FileReader r(path);
  std::vector<std::uint8_t> buf(w->nbytes);
  r.read_at(w->offset, buf.data(), buf.size());
  for (std::size_t i = 0; i < buf.size(); ++i) EXPECT_EQ(buf[i], pattern_byte(i));
}

TEST(Safetensors, AllDTypesRoundTripNameAndSize) {
  for (DType t : {DType::kBool, DType::kU8, DType::kI8, DType::kF8E4M3, DType::kF8E5M2, DType::kF8E8M0, DType::kI16, DType::kU16,
                  DType::kF16, DType::kBF16, DType::kI32, DType::kU32, DType::kF32, DType::kI64, DType::kU64, DType::kF64}) {
    DType back;
    ASSERT_TRUE(parse_dtype(dtype_name(t), back)) << dtype_name(t);
    EXPECT_EQ(back, t);
    EXPECT_GE(dtype_size(t), 1u);
  }
  DType unused;
  EXPECT_FALSE(parse_dtype("F7", unused));
}

TEST(Safetensors, RejectsBadDType) {
  TempDir dir("st_dtype");
  auto path = dir.path() / "a.safetensors";
  write_raw(path, R"({"x":{"dtype":"F7","shape":[1],"data_offsets":[0,1]}})", 1);
  try {
    SafetensorsFile f(path);
    FAIL() << "expected FormatError";
  } catch (const FormatError& e) {
    EXPECT_NE(std::string(e.what()).find("a.safetensors"), std::string::npos);
    EXPECT_NE(std::string(e.what()).find("'x'"), std::string::npos);
    EXPECT_NE(std::string(e.what()).find("F7"), std::string::npos);
  }
}

TEST(Safetensors, RejectsDuplicateName) {
  TempDir dir("st_dup");
  auto path = dir.path() / "a.safetensors";
  write_raw(path,
            R"({"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"x":{"dtype":"U8","shape":[1],"data_offsets":[1,2]}})", 2);
  EXPECT_THROW(SafetensorsFile{path}, FormatError);
}

TEST(Safetensors, RejectsOverlap) {
  TempDir dir("st_overlap");
  auto path = dir.path() / "a.safetensors";
  write_raw(path,
            R"({"a":{"dtype":"U8","shape":[4],"data_offsets":[0,4]},"b":{"dtype":"U8","shape":[4],"data_offsets":[2,6]}})", 6);
  try {
    SafetensorsFile f(path);
    FAIL() << "expected FormatError";
  } catch (const FormatError& e) {
    EXPECT_NE(std::string(e.what()).find("overlap"), std::string::npos) << e.what();
  }
}

TEST(Safetensors, RejectsGapAndTrailingBytes) {
  TempDir dir("st_gap");
  write_raw(dir.path() / "gap.safetensors",
            R"({"a":{"dtype":"U8","shape":[2],"data_offsets":[0,2]},"b":{"dtype":"U8","shape":[2],"data_offsets":[4,6]}})", 6);
  EXPECT_THROW(SafetensorsFile{dir.path() / "gap.safetensors"}, FormatError);
  write_raw(dir.path() / "tail.safetensors", R"({"a":{"dtype":"U8","shape":[2],"data_offsets":[0,2]}})", 5);
  EXPECT_THROW(SafetensorsFile{dir.path() / "tail.safetensors"}, FormatError);
  write_raw(dir.path() / "start.safetensors", R"({"a":{"dtype":"U8","shape":[2],"data_offsets":[1,3]}})", 3);
  EXPECT_THROW(SafetensorsFile{dir.path() / "start.safetensors"}, FormatError);
}

TEST(Safetensors, RejectsTruncatedData) {
  TempDir dir("st_trunc");
  auto path = dir.path() / "a.safetensors";
  write_raw(path, R"({"a":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}})", 10);
  EXPECT_THROW(SafetensorsFile{path}, FormatError);
}

TEST(Safetensors, RejectsTruncatedHeaderAndTinyFile) {
  TempDir dir("st_hdr");
  write_raw(dir.path() / "short.safetensors", "{\"a\":", 0, /*declared_header_len=*/200);
  EXPECT_THROW(SafetensorsFile{dir.path() / "short.safetensors"}, FormatError);
  std::ofstream(dir.path() / "tiny.safetensors", std::ios::binary) << "abc";
  EXPECT_THROW(SafetensorsFile{dir.path() / "tiny.safetensors"}, FormatError);
}

TEST(Safetensors, RejectsHeaderTooLarge) {
  TempDir dir("st_huge");
  auto path = dir.path() / "a.safetensors";
  write_raw(path, "{}", 0, /*declared_header_len=*/200ull * 1024 * 1024);
  EXPECT_THROW(SafetensorsFile{path}, FormatError);
  // Just over the 16 MiB cap, in a file large enough to hold it: rejected by the cap itself.
  std::string padded = "{}" + std::string(kMaxHeaderBytes - 1, ' ');
  write_raw(path, padded, 0);
  try {
    SafetensorsFile f(path);
    FAIL() << "expected FormatError";
  } catch (const FormatError& e) {
    EXPECT_NE(std::string(e.what()).find("16 MiB limit"), std::string::npos) << e.what();
  }
  // Exactly at the cap is accepted.
  padded.pop_back();
  write_raw(path, padded, 0);
  EXPECT_NO_THROW(SafetensorsFile{path});
}

TEST(Safetensors, CapsRank) {
  TempDir dir("st_rank");
  auto path = dir.path() / "a.safetensors";
  write_raw(path, R"({"a":{"dtype":"U8","shape":[1,1,1,1,1,1,1,1],"data_offsets":[0,1]}})", 1);
  EXPECT_EQ(SafetensorsFile{path}.tensors().at(0).shape.size(), kMaxRank);
  write_raw(path, R"({"a":{"dtype":"U8","shape":[1,1,1,1,1,1,1,1,1],"data_offsets":[0,1]}})", 1);
  try {
    SafetensorsFile f(path);
    FAIL() << "expected FormatError";
  } catch (const FormatError& e) {
    EXPECT_NE(std::string(e.what()).find("rank 9"), std::string::npos) << e.what();
  }
}

TEST(Safetensors, CapsTensorCount) {
  TempDir dir("st_count");
  auto path = dir.path() / "a.safetensors";
  auto header = [](std::size_t n) {
    std::string h = R"({"__metadata__":{"format":"pt"})";
    for (std::size_t i = 0; i < n; ++i) h += ",\"t" + std::to_string(i) + R"(":{"dtype":"U8","shape":[0],"data_offsets":[0,0]})";
    return h + "}";
  };
  write_raw(path, header(kMaxTensors), 0);
  EXPECT_EQ(SafetensorsFile{path}.tensors().size(), kMaxTensors);
  write_raw(path, header(kMaxTensors + 1), 0);
  try {
    SafetensorsFile f(path);
    FAIL() << "expected FormatError";
  } catch (const FormatError& e) {
    EXPECT_NE(std::string(e.what()).find("tensors"), std::string::npos) << e.what();
  }
}

TEST(Safetensors, CapsJsonValueCount) {
  // Within the byte cap but far more JSON values than any tensor table needs: rejected while
  // parsing, before the DOM grows to tens of times the header size.
  TempDir dir("st_values");
  auto path = dir.path() / "a.safetensors";
  std::string junk = R"({"__metadata__":{},"x":[)";
  for (int i = 0; i < 2'000'000; ++i) junk += "0,";
  junk += "0]}";
  ASSERT_LE(junk.size(), kMaxHeaderBytes);
  write_raw(path, junk, 0);
  try {
    SafetensorsFile f(path);
    FAIL() << "expected FormatError";
  } catch (const FormatError& e) {
    EXPECT_NE(std::string(e.what()).find("too many values"), std::string::npos) << e.what();
  }
}

TEST(Safetensors, RejectsSizeShapeMismatchAndBadFields) {
  TempDir dir("st_fields");
  auto bad = [&](const std::string& hdr, std::uint64_t data, const char* what) {
    auto path = dir.path() / "b.safetensors";
    write_raw(path, hdr, data);
    EXPECT_THROW(SafetensorsFile{path}, FormatError) << what;
  };
  bad(R"({"a":{"dtype":"F32","shape":[3],"data_offsets":[0,16]}})", 16, "byte size != numel*dtype");
  bad(R"({"a":{"dtype":"U8","shape":[-1],"data_offsets":[0,1]}})", 1, "negative dim");
  bad(R"({"a":{"dtype":"U8","shape":[1.5],"data_offsets":[0,1]}})", 1, "fractional dim");
  bad(R"({"a":{"dtype":"U8","shape":[1],"data_offsets":[1,0]}})", 1, "reversed offsets");
  bad(R"({"a":{"dtype":"U8","shape":[1],"data_offsets":[0]}})", 1, "one offset");
  bad(R"({"a":{"shape":[1],"data_offsets":[0,1]}})", 1, "missing dtype");
  bad(R"({"a":{"dtype":"U8","data_offsets":[0,1]}})", 1, "missing shape");
  bad(R"({"a":[1,2]})", 1, "entry not an object");
  bad(R"([1,2])", 0, "header not an object");
  bad(R"({"a":{"dtype":"U8","shape":[4611686018427387904,4],"data_offsets":[0,1]}})", 1, "numel overflow");
  bad("not json", 0, "invalid json");
  bad(R"({"__metadata__":{"k":1}})", 0, "non-string metadata");
}

TEST(Safetensors, ZeroTensorFileIsValid) {
  TempDir dir("st_zero");
  auto path = dir.path() / "a.safetensors";
  write_raw(path, "{}", 0);
  SafetensorsFile f(path);
  EXPECT_TRUE(f.tensors().empty());
}

// ---- FileReader / MappedFile ------------------------------------------------------------------

constexpr std::size_t kBigFile = (20u << 20) + 1234;  // not block aligned, bigger than the bounce buffer

std::filesystem::path make_pattern_file(const TempDir& dir, std::size_t n) {
  auto path = dir.path() / "data.bin";
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  std::vector<char> buf(n);
  for (std::size_t i = 0; i < n; ++i) buf[i] = static_cast<char>(pattern_byte(i));
  out.write(buf.data(), static_cast<std::streamsize>(n));
  return path;
}

void expect_pattern(const std::byte* p, std::uint64_t offset, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) {
    if (static_cast<std::uint8_t>(p[i]) != pattern_byte(offset + i)) {
      FAIL() << "mismatch at file offset " << offset + i;
    }
  }
}

TEST(FileReader, BufferedPositionalReads) {
  TempDir dir("fr_buf");
  auto path = make_pattern_file(dir, 100000);
  FileReader r(path);
  EXPECT_FALSE(r.direct());
  EXPECT_EQ(r.size(), 100000u);
  std::vector<std::byte> buf(5000);
  r.read_at(12345, buf.data(), buf.size());
  expect_pattern(buf.data(), 12345, buf.size());
  r.read_at(100000 - 7, buf.data(), 7);
  expect_pattern(buf.data(), 100000 - 7, 7);
  r.read_at(100000, buf.data(), 0);
  EXPECT_THROW(r.read_at(99999, buf.data(), 2), IoError);
  EXPECT_THROW(FileReader(dir.path() / "missing"), IoError);
}

TEST(FileReader, ConcurrentReadsShareOneHandle) {
  TempDir dir("fr_conc");
  auto path = make_pattern_file(dir, 1 << 20);
  FileReader r(path);
  std::vector<std::thread> ts;
  std::vector<int> ok(4, 0);
  for (int t = 0; t < 4; ++t) {
    ts.emplace_back([&, t] {
      std::vector<std::byte> buf(64 << 10);
      bool good = true;
      for (int k = 0; k < 20; ++k) {
        std::uint64_t off = static_cast<std::uint64_t>(t * 100000 + k * 997);
        r.read_at(off, buf.data(), buf.size());
        for (std::size_t i = 0; i < buf.size(); ++i) good = good && static_cast<std::uint8_t>(buf[i]) == pattern_byte(off + i);
      }
      ok[static_cast<std::size_t>(t)] = good;
    });
  }
  for (auto& t : ts) t.join();
  for (int v : ok) EXPECT_TRUE(v);
}

TEST(FileReader, AlignRange) {
  AlignedRange r = align_range(4096 + 10, 100);
  EXPECT_EQ(r.offset, 4096u);
  EXPECT_EQ(r.length, 4096u);
  EXPECT_EQ(r.skip, 10u);
  r = align_range(4000, 200);  // straddles a block boundary
  EXPECT_EQ(r.offset, 0u);
  EXPECT_EQ(r.length, 8192u);
  EXPECT_EQ(r.skip, 4000u);
  r = align_range(8192, 4096);
  EXPECT_EQ(r.offset, 8192u);
  EXPECT_EQ(r.length, 4096u);
  EXPECT_EQ(r.skip, 0u);
}

TEST(FileReader, AlignedBufferIsAligned) {
  AlignedBuffer b(10000);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(b.data()) % kDirectAlign, 0u);
  EXPECT_EQ(b.size(), 10000u);
}

TEST(FileReader, DirectReadsAlignedAndUnaligned) {
  TempDir dir("fr_direct");
  auto path = make_pattern_file(dir, kBigFile);
  FileReader r(path, /*direct=*/true);
  // The scratch dir is on a real file system, where O_DIRECT must be honoured; on file systems that
  // refuse it the reader falls back to buffered I/O and the correctness checks below still apply.
  std::cout << "[ INFO     ] direct I/O active: " << (r.direct() ? "yes" : "no (file system refused O_DIRECT)") << "\n";

  AlignedBuffer big(kBigFile + 2 * kDirectAlign);
  // Fully aligned request into an aligned buffer.
  r.read_at(4096, big.data(), 3 * 4096);
  expect_pattern(big.data(), 4096, 3 * 4096);
  // Unaligned offset and length into an unaligned destination.
  std::vector<std::byte> plain(70001);
  r.read_at(4096 * 5 + 17, plain.data() + 1, 70000);
  expect_pattern(plain.data() + 1, 4096 * 5 + 17, 70000);
  // Tiny read inside one block.
  r.read_at(123, plain.data(), 5);
  expect_pattern(plain.data(), 123, 5);
  // Large unaligned read spanning several bounce pieces and ending at EOF (unaligned size).
  std::vector<std::byte> huge(kBigFile - 777);
  r.read_at(777, huge.data(), huge.size());
  expect_pattern(huge.data(), 777, huge.size());
  // The last bytes of a file whose size is not a block multiple.
  r.read_at(kBigFile - 3, plain.data(), 3);
  expect_pattern(plain.data(), kBigFile - 3, 3);
  EXPECT_THROW(r.read_at(kBigFile - 1, plain.data(), 2), IoError);
}

TEST(FileReader, ReadAlignedReturnsOffsetIntoOverRead) {
  TempDir dir("fr_aligned");
  auto path = make_pattern_file(dir, 50000);
  for (bool direct : {false, true}) {
    FileReader r(path, direct);
    AlignedBuffer slot(64 * 1024);
    // Unaligned tensor-like range: 3 blocks in, 100 bytes past the boundary.
    AlignedRange range = r.read_aligned(3 * 4096 + 100, 9000, slot.data(), slot.size());
    EXPECT_EQ(range.offset, 3u * 4096);
    EXPECT_EQ(range.skip, 100u);
    expect_pattern(slot.data() + range.skip, 3 * 4096 + 100, 9000);
    // Range reaching EOF in the file's partial last block.
    range = r.read_aligned(49000, 1000, slot.data(), slot.size());
    expect_pattern(slot.data() + range.skip, 49000, 1000);
    // Buffer too small / misaligned / out of range are errors.
    EXPECT_THROW(r.read_aligned(0, 20000, slot.data(), 4096), IoError);
    EXPECT_THROW(r.read_aligned(0, 100, slot.data() + 8, slot.size() - 8), IoError);
    EXPECT_THROW(r.read_aligned(49999, 2, slot.data(), slot.size()), IoError);
  }
}

#ifdef __linux__
// Read syscalls made by this process so far (/proc/self/io "syscr", which counts pread too).
std::uint64_t read_syscalls() {
  std::ifstream io("/proc/self/io");
  std::string key;
  std::uint64_t value = 0;
  while (io >> key >> value) {
    if (key == "syscr:") return value;
  }
  return 0;
}
#endif

// A short direct read mid-file (network/FUSE file systems) may continue only when the retry is
// aligned; at or past EOF the loop must stop.
TEST(FileReader, DirectShortReadDecision) {
  const std::uint64_t file = 10 * kDirectAlign + 100;
  // EOF: position reaches the file size, aligned or not.
  EXPECT_FALSE(direct_short_read_continues(file, 0, file));
  EXPECT_FALSE(direct_short_read_continues(10 * kDirectAlign, kDirectAlign, 10 * kDirectAlign));
  EXPECT_FALSE(direct_short_read_continues(file + kDirectAlign, kDirectAlign, file));
  // Mid-file, aligned position and length: keep reading.
  EXPECT_TRUE(direct_short_read_continues(4 * kDirectAlign, 2 * kDirectAlign, file));
  EXPECT_TRUE(direct_short_read_continues(0, kDirectAlign, file));
  // Mid-file but the retry would be unaligned: stop and let the caller report it.
  EXPECT_FALSE(direct_short_read_continues(4 * kDirectAlign + 512, 2 * kDirectAlign, file));
  EXPECT_FALSE(direct_short_read_continues(4 * kDirectAlign, 2 * kDirectAlign + 1, file));
}

// The last block of a file whose size is not a multiple of kDirectAlign comes back short. The read
// loop must stop there: retrying at the unaligned EOF offset is rejected by Windows NO_BUFFERING
// handles (Linux happens to return 0 instead, so on Linux the extra syscall itself is checked).
TEST(FileReader, DirectReadOfPartialLastBlockStopsAtEof) {
  TempDir dir("fr_tail");
  for (std::size_t n : {std::size_t{100}, kDirectAlign + 1, 3 * kDirectAlign - 5}) {
    SCOPED_TRACE(n);
    auto path = make_pattern_file(dir, n);
    FileReader r(path, /*direct=*/true);
    AlignedBuffer slot(4 * kDirectAlign);
    const std::uint64_t tail = n - n % kDirectAlign;  // start of the partial last block
    AlignedRange range = r.read_aligned(tail, n - tail, slot.data(), slot.size());
    EXPECT_EQ(range.offset, tail);
    expect_pattern(slot.data() + range.skip, tail, n - tail);
    std::vector<std::byte> out(n);
    r.read_at(0, out.data(), n);  // whole file, unaligned length
    expect_pattern(out.data(), 0, n);
    r.read_at(n - 1, out.data(), 1);
    expect_pattern(out.data(), n - 1, 1);
#ifdef __linux__
    if (r.direct() && read_syscalls() != 0) {
      const std::uint64_t a = read_syscalls();
      const std::uint64_t b = read_syscalls();
      r.read_aligned(tail, n - tail, slot.data(), slot.size());
      const std::uint64_t c = read_syscalls();
      EXPECT_EQ(c - b - (b - a), 1u) << "direct read of the partial last block took more than one pread";
    }
#endif
  }
}

// An in-memory reader serves the bytes it holds exactly as the file reader serves the file (read_at, read_aligned,
// the end of file), and is never direct.
TEST(FileReader, InMemoryReadsMatchTheFile) {
  TempDir dir("fr_mem");
  const std::size_t n = 3 * kDirectAlign + 123;
  auto path = make_pattern_file(dir, n);
  std::shared_ptr<std::byte> mem(new std::byte[n], std::default_delete<std::byte[]>());
  FileReader(path).read_at(0, mem.get(), n);
  FileReader r(path, mem, n, false);
  EXPECT_FALSE(r.direct());
  EXPECT_FALSE(r.pinned());
  EXPECT_EQ(r.memory(), mem.get());
  EXPECT_EQ(r.size(), n);
  std::vector<std::byte> buf(n);
  r.read_at(17, buf.data(), 2 * kDirectAlign);
  expect_pattern(buf.data(), 17, 2 * kDirectAlign);
  AlignedBuffer slot(8 * kDirectAlign);
  const AlignedRange range = r.read_aligned(kDirectAlign + 5, kDirectAlign + 100, slot.data(), slot.size());
  expect_pattern(slot.data() + range.skip, kDirectAlign + 5, kDirectAlign + 100);
  r.read_at(n - 3, buf.data(), 3);
  expect_pattern(buf.data(), n - 3, 3);
  EXPECT_THROW(r.read_at(n - 1, buf.data(), 2), IoError);
  FileReader moved(std::move(r));
  EXPECT_EQ(moved.memory(), mem.get());
  moved.read_at(0, buf.data(), n);
  expect_pattern(buf.data(), 0, n);
}

// Whole files read into plain host memory (the soft unload's copy) hold the files' bytes, and a ReaderCache given them
// reads from memory.
TEST(FileReader, HostFilesHoldTheFiles) {
  TempDir dir("fr_hostfiles");
  const std::size_t n = 5 * kDirectAlign + 77;
  const auto a = make_pattern_file(dir, n);
  for (const bool direct : {false, true}) {
    const HostFiles hf = read_files_to_host({a, a}, /*pinned=*/false, direct, 3);
    ASSERT_EQ(hf.files.size(), 1u) << "a path named twice is read once";
    EXPECT_EQ(hf.bytes(), n);
    const HostFiles::File& f = hf.files.at(a);
    EXPECT_FALSE(f.pinned);
    ASSERT_EQ(f.size, n);
    expect_pattern(f.data.get(), 0, n);
    ModelIndex index;
    index.bundle_dir = a.parent_path();
    index.files.push_back({mi::util::path_utf8(a.filename()), n, 0});
    ReaderCache readers(index, false, &hf);
    EXPECT_EQ(readers.get(0).memory(), f.data.get());
    ReaderCache files_only(index, false);
    EXPECT_EQ(files_only.get(0).memory(), nullptr);
  }
  std::atomic<bool> cancel{true};
  EXPECT_TRUE(read_files_to_host({a}, false, false, 2, &cancel).files.empty()) << "cancelled before the first chunk";
  EXPECT_THROW(read_files_to_host({dir.path() / "missing"}, false, false), IoError);
}

// direct_io_support always explains a "no"; RAM file systems are a "no" (O_DIRECT gains nothing there, even where
// the kernel accepts the flag).
TEST(FileReader, DirectIoSupportReasons) {
  TempDir dir("fr_dsupport");
  const auto path = make_pattern_file(dir, 4 * kDirectAlign);
  const DirectIoSupport here = direct_io_support(path);
  if (here.supported) {
    EXPECT_TRUE(here.reason.empty()) << here.reason;
  } else {
    EXPECT_NE(here.reason.find(mi::util::path_utf8(path.filename())), std::string::npos) << here.reason;
  }
  const DirectIoSupport missing = direct_io_support(dir.path() / "missing.bin");
  EXPECT_FALSE(missing.supported);
  EXPECT_EQ(missing.reason, "missing.bin does not exist");
#ifdef __linux__
  if (std::filesystem::is_directory("/dev/shm")) {
    const std::filesystem::path shm = "/dev/shm/mi_test_dsupport_" + std::to_string(MI_GETPID()) + ".bin";
    std::ofstream(shm, std::ios::binary) << std::string(8192, 'x');
    const DirectIoSupport ram = direct_io_support(shm);
    std::filesystem::remove(shm);
    EXPECT_FALSE(ram.supported);
    EXPECT_FALSE(ram.reason.empty());
  }
#endif
}

// The page-cache residency probe, both queries (cachestat, and the mincore sample a kernel without cachestat gets): a
// file just written sits in the page cache, the same file written back and dropped from it does not; a missing file,
// or the seam's "no query", is unknown (negative). Windows has no query (negative).
TEST(FileReader, PageCacheFraction) {
  TempDir dir("fr_pagecache");
  const auto path = make_pattern_file(dir, 256 * kDirectAlign + 5);
  namespace seams = mi::weights::testing;
  using seams::ResidencyQuery;
  struct Restore {
    ~Restore() { seams::force_residency_query(ResidencyQuery::kAuto); }
  } restore;
#ifdef __linux__
  for (const ResidencyQuery q : {ResidencyQuery::kAuto, ResidencyQuery::kMincore}) {
    SCOPED_TRACE(q == ResidencyQuery::kAuto ? "auto" : "mincore");
    seams::force_residency_query(q);
    {
      std::ifstream warm(path, std::ios::binary);
      std::string all((std::istreambuf_iterator<char>(warm)), std::istreambuf_iterator<char>());
    }
    const double cached = page_cache_fraction(path);
    EXPECT_GE(cached, 0.9) << "a file just read is in the page cache";
    EXPECT_LE(cached, 1.0);
    struct statfs fsinfo {};
    ASSERT_EQ(::statfs(path.c_str(), &fsinfo), 0);
    const auto type = static_cast<std::uint32_t>(fsinfo.f_type);
    if (type == kTmpfsMagic || type == kRamfsMagic) continue;  // the page cache is the storage: nothing to drop
    const int fd = ::open(path.c_str(), O_RDONLY);
    ASSERT_GE(fd, 0);
    ::fsync(fd);
    ::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    ::close(fd);
    EXPECT_LT(page_cache_fraction(path), 0.9) << "written back and dropped from the page cache";
  }
#else
  EXPECT_LT(page_cache_fraction(path), 0.0);
#endif
#ifdef __linux__
  // A file the process may not write (root's, for a user): mincore would claim every page resident, cachestat refuses
  // it; both queries say "unknown".
  if (::geteuid() != 0 && std::filesystem::is_regular_file("/etc/os-release")) {
    for (const ResidencyQuery q : {ResidencyQuery::kAuto, ResidencyQuery::kMincore}) {
      seams::force_residency_query(q);
      EXPECT_LT(page_cache_fraction("/etc/os-release"), 0.0);
    }
  }
#endif
  seams::force_residency_query(ResidencyQuery::kNone);
  EXPECT_LT(page_cache_fraction(path), 0.0);
  seams::force_residency_query(ResidencyQuery::kAuto);
  EXPECT_LT(page_cache_fraction(dir.path() / "missing.bin"), 0.0);
}

TEST(MappedFile, ViewMatchesFileContents) {
  TempDir dir("mm");
  auto path = make_pattern_file(dir, 300000);
  MappedFile m(path);
  ASSERT_EQ(m.size(), 300000u);
  expect_pattern(m.data(), 0, m.size());
  MappedFile moved(std::move(m));
  EXPECT_EQ(moved.size(), 300000u);
  EXPECT_EQ(m.data(), nullptr);  // NOLINT: moved-from state is specified
  expect_pattern(moved.data() + 1000, 1000, 500);
}

// check_unchanged, through the descriptor the map keeps: quiet while the file is as mapped (a moved map too); a file
// written since (its modification time moved) or, on POSIX, truncated below the end the caller reads to is an IoError
// before the map is read, never a SIGBUS.
TEST(MappedFile, CheckUnchangedSeesWritesAndTruncation) {
  TempDir dir("mm_check");
  const auto path = make_pattern_file(dir, 300000);
  {
    MappedFile m(path);
    EXPECT_NO_THROW(m.check_unchanged(300000));
    std::filesystem::last_write_time(path, std::filesystem::last_write_time(path) + std::chrono::seconds(5));
    EXPECT_THROW(m.check_unchanged(300000), IoError);
  }
#ifndef _WIN32  // Windows refuses to truncate a mapped file
  MappedFile m(path);
  const MappedFile moved(std::move(m));
  EXPECT_NO_THROW(moved.check_unchanged(300000));
  std::filesystem::resize_file(path, 1000);
  try {
    moved.check_unchanged(300000);
    ADD_FAILURE() << "a truncated map passed";
  } catch (const IoError& e) {
    EXPECT_EQ(std::string(e.what()), mi::util::path_utf8(path) + ": the file changed while it was mapped; load it again");
  }
#endif
}

TEST(MappedFile, EmptyAndMissingFiles) {
  TempDir dir("mm_empty");
  std::ofstream(dir.path() / "e.bin", std::ios::binary).close();
  MappedFile m(dir.path() / "e.bin");
  EXPECT_EQ(m.size(), 0u);
  EXPECT_THROW(MappedFile(dir.path() / "missing.bin"), IoError);
}

#ifndef _WIN32
// A FIFO given as a weight file is refused at once (an open(2) of one blocks until a writer appears), buffered and
// direct: by FileReader (so SafetensorsFile), MappedFile, direct_io_support, page_cache_fraction and
// read_replaceable_file (caches). error_within fails the test instead of hanging on a regression.
TEST(FileReader, RefusesFifoPromptly) {
  TempDir dir("fr_fifo");
  const auto fifo = dir.path() / "model.safetensors";
  ASSERT_EQ(::mkfifo(fifo.c_str(), 0600), 0) << std::strerror(errno);
  const std::string not_regular = mi::util::path_utf8(fifo) + ": not a regular file";
  for (const bool direct : {false, true}) {
    const std::optional<std::string> e = mi::testing::error_within(fifo, [&] { FileReader r(fifo, direct); });
    ASSERT_TRUE(e.has_value()) << "FileReader blocked on a FIFO (direct " << direct << ")";
    EXPECT_EQ(*e, not_regular) << direct;
  }
  std::optional<std::string> e = mi::testing::error_within(fifo, [&] { SafetensorsFile f(fifo); });
  ASSERT_TRUE(e.has_value()) << "SafetensorsFile blocked on a FIFO";
  EXPECT_EQ(*e, not_regular);
  e = mi::testing::error_within(fifo, [&] { MappedFile m(fifo); });
  ASSERT_TRUE(e.has_value()) << "MappedFile blocked on a FIFO";
  EXPECT_EQ(*e, not_regular);
  e = mi::testing::error_within(fifo, [&] {
    const DirectIoSupport d = direct_io_support(fifo);
    EXPECT_FALSE(d.supported);
    EXPECT_EQ(d.reason, "model.safetensors is not a regular file");
    EXPECT_LT(page_cache_fraction(fifo), 0.0);
    std::string text;
    EXPECT_EQ(mi::util::read_replaceable_file(fifo, 1024, text), mi::util::ReadStatus::kError);
  });
  ASSERT_TRUE(e.has_value()) << "a FIFO probe blocked";
  EXPECT_EQ(*e, "");
}
#endif

// A directory is not a file either (refused by its type before any open); a regular file, or a symlink to one, still
// opens through the same checks, with and without direct I/O.
TEST(FileReader, RefusesDirectoryAndReadsRegularFiles) {
  TempDir dir("fr_dir");
  try {
    FileReader r(dir.path());
    ADD_FAILURE() << "a directory opened as a file";
  } catch (const IoError& e) {
#ifndef _WIN32
    EXPECT_EQ(std::string(e.what()), mi::util::path_utf8(dir.path()) + ": not a regular file");
#endif
  }
  EXPECT_THROW(MappedFile m(dir.path()), IoError);
  EXPECT_THROW(SafetensorsFile f(dir.path()), IoError);
  EXPECT_FALSE(direct_io_support(dir.path()).supported);
  std::string text;
  EXPECT_EQ(mi::util::read_replaceable_file(dir.path(), 1024, text), mi::util::ReadStatus::kError);
  const auto path = make_pattern_file(dir, 3 * kDirectAlign + 5);
  for (const bool direct : {false, true}) {
    FileReader r(path, direct);
    EXPECT_EQ(r.size(), 3 * kDirectAlign + 5);
    std::vector<std::byte> buf(kDirectAlign + 9);
    r.read_at(kDirectAlign - 4, buf.data(), buf.size());
    expect_pattern(buf.data(), kDirectAlign - 4, buf.size());
  }
  // A symlink to a regular file opens as the file: the type checks follow it.
  const auto link = dir.path() / "link.bin";
  std::error_code ec;
  std::filesystem::create_symlink(path, link, ec);
#ifndef _WIN32
  ASSERT_FALSE(ec) << ec.message();
#endif
  if (!ec) {  // Windows creates symlinks only with a privilege or developer mode
    for (const bool direct : {false, true}) {
      FileReader r(link, direct);
      EXPECT_EQ(r.size(), 3 * kDirectAlign + 5);
      std::vector<std::byte> buf(100);
      r.read_at(2 * kDirectAlign + 1, buf.data(), buf.size());
      expect_pattern(buf.data(), 2 * kDirectAlign + 1, buf.size());
    }
    EXPECT_EQ(MappedFile(link).size(), 3 * kDirectAlign + 5);
  }
  ASSERT_EQ(mi::util::read_replaceable_file(path, 1 << 20, text), mi::util::ReadStatus::kOk);
  EXPECT_EQ(text.size(), 3 * kDirectAlign + 5);
  expect_pattern(reinterpret_cast<const std::byte*>(text.data()), 0, text.size());
  EXPECT_EQ(mi::util::read_replaceable_file(path, 3 * kDirectAlign + 4, text), mi::util::ReadStatus::kTooLarge);
  EXPECT_EQ(mi::util::read_replaceable_file(dir.path() / "missing", 1024, text), mi::util::ReadStatus::kMissing);
}

// The Windows device-path rules (util::windows_path_names_device), pure, so checked on every platform: the device and
// NT namespaces name a file only as a drive path or a UNC share; UNC pipe and mailslot shares are devices on any host;
// DOS device names are caught in their GetFullPathNameW form (\\.\NAME). Case variants and both separators.
TEST(RegularFile, WindowsDevicePaths) {
  using mi::util::windows_path_names_device;
  const wchar_t* devices[] = {
      LR"(\\.\pipe\x)", LR"(\\?\pipe\x)", LR"(//?/pipe/x)", LR"(//./pipe/x)", LR"(\??\pipe\x)", LR"(/??/pipe/x)",
      LR"(\\.\COM1)", LR"(\\?\COM1)", LR"(\\.\PhysicalDrive0)", LR"(\\?\physicaldrive0)", LR"(\\.\NUL)",
      LR"(\\.\CONIN$)", LR"(\\.\C:)", LR"(\\?\c:)", LR"(\\?\GLOBALROOT\Device\Null)",
      LR"(\\?\globalroot\device\harddiskvolume1\m.safetensors)", LR"(\\?\GlobalRoot\x)",
      LR"(\\?\Volume{01234567-89ab-cdef-0123-456789abcdef}\m.safetensors)", LR"(\??\GLOBALROOT\x)",
      LR"(\/?\pipe\x)", LR"(/\.\COM1)", LR"(\\localhost\pipe\x)", LR"(\\host\PIPE\x)", LR"(//host/Pipe/x)",
      LR"(\\host\mailslot\x)", LR"(\\host\MAILSLOT\x)", LR"(\\?\UNC\host\pipe\x)", LR"(\\?\unc\host\Pipe)",
      LR"(//?/UNC/host/mailslot/x)", LR"(\??\UNC\localhost\pipe\x)", LR"(\\?\UNC\host)", LR"(\\?\UNC)",
      LR"(\\host)", LR"(\\host\)", LR"(\\.)", LR"(\\?)",
      // Provider-qualified UNC (\\;Provider\server\share) names a redirector; the NT upcase of dotless i and long s is
      // ASCII, so these share names are pipe, IPC$ and mailslot.
      LR"(\\;LanmanRedirector\host\pipe\x)", LR"(\\;x\host\share\m.safetensors)", LR"(\\?\UNC\;Lanman\host\share\x)",
      L"\\\\host\\p\u0131pe\\x", L"\\\\host\\\u0131PC$\\x", L"\\\\host\\mail\u017Flot\\x",
      // IPC$ (the named pipes' share), and share names Win32 normalization turns into pipe (trailing dots, spaces).
      LR"(\\host\IPC$\x)", LR"(\\?\UNC\host\ipc$\x)", LR"(//host/Ipc$/x)", LR"(\\host\pipe.\x)",
      LR"(\\host\pipe..\x)", LR"(\\host\pipe \x)", LR"(\\host\mailslot.\x)", LR"(\\?\UNC\host\pipe.\x)",
      // DOS device names as GetFullPathNameW gives them, superscript-digit ports included (COM¹ is COM1 to Windows).
      L"\\\\.\\COM\u00B9", L"\\\\.\\COM\u00B2", L"\\\\.\\COM\u00B3", L"\\\\.\\LPT\u00B9", LR"(\\.\LPT1)", LR"(\\.\AUX)",
      LR"(\\.\PRN)", LR"(\\.\CON)"};
  for (const wchar_t* p : devices) EXPECT_TRUE(windows_path_names_device(p)) << std::wstring(p).size() << " chars";
  const wchar_t* files[] = {
      LR"(C:\m\model.safetensors)", LR"(c:/m/model.safetensors)", LR"(m\model.safetensors)", LR"(model.safetensors)",
      LR"(\m\model.safetensors)", LR"(\\?\C:\m\model.safetensors)", LR"(\\?\c:\)", LR"(//?/C:/m/x)",
      LR"(\\.\D:\m\x)", LR"(\??\C:\m\x)", LR"(\\nas\models\m.safetensors)", LR"(//nas/models/m.safetensors)",
      LR"(\\nas\pipeline\m.safetensors)", LR"(\\nas\mailslots\m)", LR"(\\?\UNC\nas\models\m.safetensors)",
      LR"(\\?\unc\nas\share\x)", LR"(//?/UNC/nas/models/m)", LR"(C:\m\COM1)", LR"(COM1.txt)", LR"(NUL)",
      LR"(C:\pipe\x)", LR"(pipe\x)",
      // `pipe$` is an ordinary (hidden) file share, not the named pipes' namespace (that is `pipe`, or IPC$).
      LR"(\\host\pipe$\x)", LR"(\\?\UNC\host\pipe$\x)", LR"(\\host\pipes\x)", LR"(\\host\ipc\x)",
      // 8.3 short names.
      LR"(C:\PROGRA~1\MOSSIN~1\MODEL~1.SAF)", LR"(\\nas\MODELS~1\X~1.SAF)", LR"(\\?\C:\PROGRA~1\M~1)",
      // Text that only GetFullPathNameW turns into a device (on Windows): not a device by its text.
      L"C:\\m\\COM\u00B9", L"COM\u00B9.txt", LR"(CONIN$)"};
  for (const wchar_t* p : files) EXPECT_FALSE(windows_path_names_device(p)) << std::wstring(p).size() << " chars";
  // Long paths under \\?\ (beyond MAX_PATH, up to the 32767 characters NT allows) are files.
  std::wstring long_path = LR"(\\?\C:\models)";
  while (long_path.size() < 32000) long_path += L"\\a-rather-long-folder-name";
  EXPECT_FALSE(windows_path_names_device(long_path + L"\\model.safetensors"));
  EXPECT_FALSE(windows_path_names_device(LR"(\\?\UNC\nas\models)" + long_path.substr(13)));
  EXPECT_TRUE(windows_path_names_device(LR"(\\?\UNC\nas\pipe)" + long_path.substr(13)));
}

// The boundary check of user paths (util::check_user_path): on Windows a device or pipe path is refused, naming the
// field, before any file-system call; off Windows it is a no-op (device files are refused where they are opened, by
// util::open_regular_file). util::path_key opens nothing on Windows and is weakly_canonical elsewhere.
TEST(RegularFile, UserPathBoundary) {
  TempDir dir("user_path");
  const auto file = make_pattern_file(dir, 16);
  EXPECT_NO_THROW(mi::util::check_user_path(file, "dit_file"));
  EXPECT_NO_THROW(mi::util::check_user_path(dir.path(), "cache_dir"));
  EXPECT_NO_THROW(mi::util::check_user_path({}, "cache_dir"));  // an unset option is not a path
#ifdef _WIN32
  for (const char* p : {R"(\\.\pipe\x)", R"(\\?\GLOBALROOT\Device\Null)", R"(\\host\IPC$\x)", R"(//./COM1)", "NUL"}) {
    try {
      mi::util::check_user_path(mi::util::path_from_utf8(p), "dit_file");
      ADD_FAILURE() << p << " passed";
    } catch (const std::invalid_argument& e) {
      EXPECT_EQ(std::string(e.what()), std::string("dit_file: not a regular file: ") + p);
    }
  }
  EXPECT_EQ(mi::util::path_key(dir.path() / "sub" / ".." / "data.bin"), std::filesystem::absolute(file));
#else
  EXPECT_NO_THROW(mi::util::check_user_path(R"(\\.\pipe\x)", "dit_file"));
  EXPECT_NO_THROW(mi::util::check_user_path("/dev/null", "dit_file"));
  EXPECT_EQ(mi::util::path_key(dir.path() / "sub" / ".." / "data.bin"), std::filesystem::weakly_canonical(file));
#endif
}

// util::file_info: one query that follows links, giving a regular file's size, its last write time (the value
// std::filesystem::last_write_time converts to, as .mossidx files persist it) and its identity (a hard link shares it,
// another file does not); a directory, a missing path, and on Linux a FIFO (kOther, never opened). A file replaced
// under its path, or written in place, gets another stamp. util::resolved_path resolves as weakly_canonical does.
// util::OutputFile writes, seeks past the end (the gap reads as zeros) and refuses a directory.
TEST(RegularFile, FileInfoAndOutputFile) {
  namespace fs = std::filesystem;
  using mi::util::FileKind;
  using mi::util::file_info;
  TempDir dir("file_info");
  const auto file = make_pattern_file(dir, 1000);
  const mi::util::FileInfo info = file_info(file);
  EXPECT_EQ(info.kind, FileKind::kRegular);
  EXPECT_EQ(info.size, 1000u);
#ifndef _WIN32  // Windows: WindowsFileInfoFromHandle checks the time against the handle's FILETIME
  const auto t = fs::last_write_time(file);
#if defined(_GLIBCXX_RELEASE) && _GLIBCXX_RELEASE < 13
  const auto sys = fs::file_time_type::clock::to_sys(t);
#else
  const auto sys = std::chrono::clock_cast<std::chrono::system_clock>(t);
#endif
  EXPECT_EQ(info.mtime_ns, std::chrono::duration_cast<std::chrono::nanoseconds>(sys.time_since_epoch()).count());
#endif
  EXPECT_EQ(file_info(file), info);
  fs::create_hard_link(file, dir.path() / "link.bin");
  EXPECT_TRUE(mi::util::same_file(file_info(dir.path() / "link.bin"), info));
  const fs::path other = dir.path() / "other.bin";
  { std::ofstream(other, std::ios::binary) << "x"; }
  EXPECT_FALSE(mi::util::same_file(file_info(other), info));
  EXPECT_EQ(file_info(dir.path()).kind, FileKind::kDirectory);
  const mi::util::FileInfo missing = file_info(dir.path() / "missing");
  EXPECT_EQ(missing.kind, FileKind::kMissing);
  EXPECT_FALSE(mi::util::same_file(missing, missing));
  EXPECT_EQ(file_info({}).kind, FileKind::kMissing);
  fs::rename(other, file);  // replaced under the same path
  const mi::util::FileInfo replaced = file_info(file);
  EXPECT_NE(replaced, info);
  EXPECT_EQ(replaced.size, 1u);
  { std::ofstream(file, std::ios::binary | std::ios::app) << "yz"; }  // written in place
  EXPECT_NE(file_info(file), replaced);
  EXPECT_EQ(mi::util::resolved_path(dir.path() / "sub" / ".." / "data.bin"), mi::util::resolved_path(file));
#ifdef __linux__
  ASSERT_EQ(::mkfifo((dir.path() / "fifo").c_str(), 0600), 0);
  EXPECT_EQ(file_info(dir.path() / "fifo").kind, FileKind::kOther);
#endif

  const fs::path out_path = dir.path() / "out.bin";
  {
    mi::util::OutputFile out(out_path);
    ASSERT_TRUE(out.is_open());
    EXPECT_TRUE(out.write("abc", 3));
    EXPECT_TRUE(out.seek(6));
    EXPECT_TRUE(out.write(std::string_view("de")));
    EXPECT_TRUE(out.seek(0));
    EXPECT_TRUE(out.write("A", 1));
    EXPECT_TRUE(out.close());
  }
  std::string text;
  ASSERT_EQ(mi::util::read_replaceable_file(out_path, 64, text), mi::util::ReadStatus::kOk);
  EXPECT_EQ(text, std::string("Abc\0\0\0de", 8));
  mi::util::OutputFile refused(dir.path());
  EXPECT_FALSE(refused.is_open());
  EXPECT_FALSE(refused.write("x", 1));
  EXPECT_FALSE(refused.close());
#ifdef __linux__
  // An output naming a FIFO is refused unopened (without a reader, an open for writing would block).
  EXPECT_FALSE(mi::util::OutputFile(dir.path() / "fifo").is_open());
#endif
  // Temporaries: a new name beside the target each time ("<target>.tmp-<hex>"), created exclusively; synced and
  // closed through the same descriptor.
  using Create = mi::util::OutputFile::Create;
  mi::util::OutputFile t1(out_path, Create::kTemporary), t2(out_path, Create::kTemporary);
  ASSERT_TRUE(t1.is_open() && t2.is_open());
  EXPECT_NE(t1.path(), t2.path());
  EXPECT_EQ(t1.path().parent_path(), out_path.parent_path());
  EXPECT_EQ(mi::util::path_utf8(t1.path().filename()).rfind("out.bin.tmp-", 0), 0u);
  EXPECT_TRUE(t1.write("x", 1));
  EXPECT_TRUE(t1.sync());
  EXPECT_TRUE(t1.close());
  EXPECT_EQ(file_info(t1.path()).size, 1u);
  EXPECT_TRUE(mi::testing::temporary_left(out_path));
  // publish renames the temporary over its target; discard deletes it.
  EXPECT_TRUE(t2.write("pub", 3));
  EXPECT_TRUE(t2.publish(out_path));
  EXPECT_EQ(t2.path(), out_path);
  ASSERT_EQ(mi::util::read_replaceable_file(out_path, 64, text), mi::util::ReadStatus::kOk);
  EXPECT_EQ(text, "pub");
  mi::util::OutputFile t3(out_path, Create::kTemporary);
  const fs::path t3_path = t3.path();
  t3.discard();
  EXPECT_FALSE(file_info(t3_path).exists());
}

// list_directory: every entry but . and .., with its own type, size and time from the listing (a link not followed);
// touch_file_in and remove_file_in act on a plain file in the folder (on Windows only there: a link or a file
// elsewhere is refused).
TEST(RegularFile, ListTouchAndRemoveInFolder) {
  namespace fs = std::filesystem;
  TempDir dir("list_dir");
  const auto file = make_pattern_file(dir, 1234);
  fs::create_directory(dir.path() / "sub");
  std::error_code ec;
  fs::create_symlink(file, dir.path() / "link.bin", ec);  // Windows: only with the privilege or developer mode
  std::vector<mi::util::DirEntry> entries;
  ASSERT_TRUE(mi::util::list_directory(dir.path(), entries));
  std::map<std::string, mi::util::DirEntry> by_name;
  for (const auto& e : entries) by_name[mi::util::path_utf8(e.name)] = e;
  ASSERT_EQ(by_name.size(), ec ? 2u : 3u);
  EXPECT_EQ(by_name["data.bin"].info.kind, mi::util::FileKind::kRegular);
  EXPECT_EQ(by_name["data.bin"].info.size, 1234u);
  EXPECT_EQ(by_name["data.bin"].info.mtime_ns, mi::util::file_info(file).mtime_ns);
  EXPECT_FALSE(by_name["data.bin"].link);
  EXPECT_EQ(by_name["sub"].info.kind, mi::util::FileKind::kDirectory);
  if (!ec) {
    EXPECT_TRUE(by_name["link.bin"].link);
    EXPECT_FALSE(by_name["link.bin"].info.regular());
  }
  EXPECT_FALSE(mi::util::list_directory(dir.path() / "missing", entries));
  EXPECT_FALSE(mi::util::list_directory(file, entries));

  const fs::path folder = mi::util::resolved_path(dir.path());
  const std::int64_t before = mi::util::file_info(file).mtime_ns;
  fs::last_write_time(file, fs::last_write_time(file) - std::chrono::hours(1));
  ASSERT_TRUE(mi::util::touch_file_in(folder, file));
  EXPECT_GE(mi::util::file_info(file).mtime_ns, before);
#ifdef _WIN32
  if (!ec) EXPECT_FALSE(mi::util::remove_file_in(folder, dir.path() / "link.bin"));  // a link is never acted on
  EXPECT_FALSE(mi::util::remove_file_in(mi::util::resolved_path(dir.path() / "sub"), file));  // nor a file elsewhere
#endif
  EXPECT_TRUE(mi::util::remove_file_in(folder, file));
  EXPECT_FALSE(mi::util::file_info(file).exists());
  EXPECT_FALSE(mi::util::remove_file_in(folder, file));

  // make_directories: the missing levels only; an existing folder is fine, a file in the way refuses the whole.
  EXPECT_TRUE(mi::util::make_directories(dir.path() / "a" / "b" / "c"));
  EXPECT_TRUE(mi::util::file_info(dir.path() / "a" / "b" / "c").directory());
  EXPECT_TRUE(mi::util::make_directories(dir.path() / "a" / "b"));
  std::ofstream(dir.path() / "a" / "plain") << "x";
  EXPECT_FALSE(mi::util::make_directories(dir.path() / "a" / "plain" / "d"));
  EXPECT_FALSE(mi::util::make_directories(dir.path() / "a" / "plain"));
#ifdef __linux__
  ASSERT_EQ(::mkfifo((dir.path() / "fifo").c_str(), 0600), 0);
  EXPECT_FALSE(mi::util::make_directories(dir.path() / "fifo" / "d"));
#endif
}

#ifdef _WIN32
namespace {

// Turns the empty directory `dir` into a junction (a mount point) to the NT path `target` (e.g. \??\pipe\name), as an
// unprivileged user can in any folder they may write; false when the file system refuses.
bool make_junction(const std::filesystem::path& dir, const std::wstring& target) {
  HANDLE h = CreateFileW(dir.c_str(), GENERIC_WRITE, 0, nullptr, OPEN_EXISTING,
                         FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
  if (h == INVALID_HANDLE_VALUE) return false;
  // REPARSE_DATA_BUFFER (ntifs.h) with its MountPointReparseBuffer: tag, data length, reserved, the substitute name's
  // offset and length, the print name's, then the substitute name and an empty print name, each NUL-terminated.
  std::vector<std::uint8_t> buf(16 + (target.size() + 2) * sizeof(wchar_t));
  const auto put16 = [&](std::size_t at, std::size_t v) {
    const auto x = static_cast<std::uint16_t>(v);
    std::memcpy(buf.data() + at, &x, sizeof x);
  };
  const DWORD tag = IO_REPARSE_TAG_MOUNT_POINT;
  std::memcpy(buf.data(), &tag, sizeof tag);
  put16(4, buf.size() - 8);
  put16(8, 0);
  put16(10, target.size() * sizeof(wchar_t));
  put16(12, (target.size() + 1) * sizeof(wchar_t));
  put16(14, 0);
  std::memcpy(buf.data() + 16, target.data(), target.size() * sizeof(wchar_t));
  DWORD returned = 0;
  const bool ok = DeviceIoControl(h, FSCTL_SET_REPARSE_POINT, buf.data(), static_cast<DWORD>(buf.size()), nullptr, 0,
                                  &returned, nullptr) != 0;
  CloseHandle(h);
  return ok;
}

// One instance of the named pipe \\.\pipe\<name> with an overlapped ConnectNamedPipe pending: connected() says,
// without waiting, whether a client open reached it; the server side then reads what the client wrote and impersonates
// it.
class PipeInstance {
 public:
  PipeInstance(const std::wstring& name, bool first) {
    event_ = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    ov_.hEvent = event_;
    const DWORD open = PIPE_ACCESS_DUPLEX | FILE_FLAG_OVERLAPPED | (first ? FILE_FLAG_FIRST_PIPE_INSTANCE : 0);
    pipe_ = CreateNamedPipeW((L"\\\\.\\pipe\\" + name).c_str(), open, PIPE_TYPE_BYTE | PIPE_WAIT,
                             PIPE_UNLIMITED_INSTANCES, 4096, 4096, 0, nullptr);
    if (pipe_ != INVALID_HANDLE_VALUE && !ConnectNamedPipe(pipe_, &ov_) && GetLastError() == ERROR_PIPE_CONNECTED) {
      SetEvent(event_);
    }
  }
  ~PipeInstance() {
    if (pipe_ != INVALID_HANDLE_VALUE) {
      CancelIo(pipe_);
      CloseHandle(pipe_);
    }
    if (event_ != nullptr) CloseHandle(event_);
  }
  PipeInstance(const PipeInstance&) = delete;
  PipeInstance& operator=(const PipeInstance&) = delete;

  bool ok() const { return pipe_ != INVALID_HANDLE_VALUE && event_ != nullptr; }
  bool connected() const { return WaitForSingleObject(event_, 0) == WAIT_OBJECT_0; }
  // Reads the byte the client wrote (a pipe server may impersonate only after a read), then impersonates the client:
  // its token's SECURITY_IMPERSONATION_LEVEL, or -1 when ImpersonateNamedPipeClient refuses.
  int read_and_impersonate() {
    char byte = 0;
    DWORD n = 0;
    ResetEvent(event_);
    if (!ReadFile(pipe_, &byte, 1, nullptr, &ov_) && GetLastError() != ERROR_IO_PENDING) return -1;
    if (!GetOverlappedResult(pipe_, &ov_, &n, TRUE) || n != 1) return -1;
    return impersonate();
  }
  int impersonate() {
    if (!ImpersonateNamedPipeClient(pipe_)) return -1;
    HANDLE token = nullptr;
    SECURITY_IMPERSONATION_LEVEL level = SecurityAnonymous;
    DWORD n = 0;
    const bool queried = OpenThreadToken(GetCurrentThread(), TOKEN_QUERY, TRUE, &token) &&
                         GetTokenInformation(token, TokenImpersonationLevel, &level, sizeof level, &n);
    if (token != nullptr) CloseHandle(token);
    RevertToSelf();
    return queried ? static_cast<int>(level) : -2;
  }

 private:
  HANDLE pipe_ = INVALID_HANDLE_VALUE;
  HANDLE event_ = nullptr;
  OVERLAPPED ov_{};
};

std::vector<std::unique_ptr<PipeInstance>> pipe_instances(const std::wstring& name, int count) {
  std::vector<std::unique_ptr<PipeInstance>> out;
  for (int i = 0; i < count; ++i) out.push_back(std::make_unique<PipeInstance>(name, i == 0));
  return out;
}

int connected_count(const std::vector<std::unique_ptr<PipeInstance>>& pipes) {
  int n = 0;
  for (const auto& p : pipes) n += p->connected() ? 1 : 0;
  return n;
}

}  // namespace

// The design limit of check_user_path (it classifies the text only): a junction inside a folder the engine reads can
// lead an ordinary-looking path to a named pipe, with a server listening. Every helper refuses it and none blocks:
//   - a junction to the pipe namespace (\??\pipe\) with the pipe's name below it (a cache folder redirected) reaches
//     the pipe: file_info says kOther, open_regular_file kNotRegular, OutputFile and FileReader refuse, and whatever
//     the server gets from those opens impersonates at SecurityIdentification at most;
//   - a junction that is itself the final component, to \??\pipe\<name>: Windows does not resolve a final-component
//     mount point to a pipe (the open fails as not found), and the helpers fail closed: never a regular file.
TEST(RegularFile, WindowsJunctionToPipeIsRefused) {
  TempDir dir("junction_pipe");
  const std::wstring name = L"mi_test_pipe_" + std::to_wstring(GetCurrentProcessId());
  const auto pipes = pipe_instances(name, 16);  // enough for every client open below
  for (const auto& p : pipes) ASSERT_TRUE(p->ok()) << GetLastError();
  const std::filesystem::path to_pipe = dir.path() / "model.safetensors";
  const std::filesystem::path to_namespace = dir.path() / "cache";
  std::filesystem::create_directory(to_pipe);
  std::filesystem::create_directory(to_namespace);
  if (!make_junction(to_pipe, L"\\??\\pipe\\" + name) || !make_junction(to_namespace, L"\\??\\pipe\\")) {
    GTEST_SKIP() << "this file system refuses junctions (" << GetLastError() << ")";
  }
  const std::filesystem::path in_namespace = to_namespace / name;
  for (const std::filesystem::path& p : {in_namespace, to_pipe}) {
    SCOPED_TRACE(p.string());
    const auto t0 = std::chrono::steady_clock::now();
    EXPECT_NO_THROW(mi::util::check_user_path(p, "dit_file"));  // the text is an ordinary path
    const mi::util::FileInfo info = mi::util::file_info(p);
    EXPECT_FALSE(info.regular() || info.directory());
    void* handle = nullptr;
    EXPECT_NE(mi::util::open_regular_file(p, FILE_ATTRIBUTE_NORMAL, handle), mi::util::OpenStatus::kOk);
    EXPECT_FALSE(mi::util::OutputFile(p).is_open());
    EXPECT_THROW(FileReader{p}, IoError);
    EXPECT_LT(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), 5.0);
    if (p == in_namespace) {
      EXPECT_EQ(info.kind, mi::util::FileKind::kOther);
      EXPECT_GT(connected_count(pipes), 0) << "the opens did not reach the pipe";
    } else {
      EXPECT_NE(info.kind, mi::util::FileKind::kRegular);
    }
  }
  // The engine's opens write nothing, so the server cannot even read before impersonating; were impersonation
  // allowed, it would be at identification level.
  for (const auto& p : pipes) {
    if (!p->connected()) continue;
    const int level = p->impersonate();
    EXPECT_TRUE(level == -1 || level == SecurityIdentification) << level;
  }
}

// What the server of a pipe gets from a client that opens it with the engine's flags (util::kUserPathSqos, on every
// CreateFileW of a user path) and writes: ImpersonateNamedPipeClient succeeds, but the token is at
// SecurityIdentification, so the server can learn who the engine runs as and cannot act as it. The same client without
// SQOS gives SecurityImpersonation: the test can tell the two apart. With every instance busy, file_info returns at
// once (kOther from ERROR_PIPE_BUSY), never waiting for an instance.
TEST(RegularFile, WindowsPipeOpenIsIdentificationOnlyAndNeverBlocks) {
  const std::wstring name = L"mi_test_sqos_" + std::to_wstring(GetCurrentProcessId());
  const std::wstring path = L"\\\\.\\pipe\\" + name;
  const auto level_of = [&](DWORD flags) {
    PipeInstance server(name, true);
    if (!server.ok()) return -3;
    HANDLE client = CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_EXISTING, flags, nullptr);
    if (client == INVALID_HANDLE_VALUE) return -4;
    const char byte = 1;
    DWORD n = 0;
    const bool wrote = WriteFile(client, &byte, 1, &n, nullptr) && n == 1;
    const int level = wrote ? server.read_and_impersonate() : -5;
    CloseHandle(client);
    return level;
  };
  EXPECT_EQ(level_of(FILE_ATTRIBUTE_NORMAL | mi::util::kUserPathSqos), SecurityIdentification);
  EXPECT_EQ(level_of(FILE_ATTRIBUTE_NORMAL), SecurityImpersonation);

  TempDir dir("pipe_busy");
  const std::filesystem::path to_namespace = dir.path() / "cache";
  std::filesystem::create_directory(to_namespace);
  if (!make_junction(to_namespace, L"\\??\\pipe\\")) GTEST_SKIP() << "this file system refuses junctions";
  PipeInstance only(name, true);
  ASSERT_TRUE(only.ok());
  HANDLE holder = CreateFileW(path.c_str(), GENERIC_READ, 0, nullptr, OPEN_EXISTING, mi::util::kUserPathSqos, nullptr);
  ASSERT_NE(holder, INVALID_HANDLE_VALUE) << GetLastError();
  const auto t0 = std::chrono::steady_clock::now();
  EXPECT_EQ(mi::util::file_info(to_namespace / name).kind, mi::util::FileKind::kOther);
  void* handle = nullptr;
  EXPECT_NE(mi::util::open_regular_file(to_namespace / name, FILE_ATTRIBUTE_NORMAL, handle), mi::util::OpenStatus::kOk);
  EXPECT_LT(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), 5.0);
  CloseHandle(holder);
}

// A regular file's info from its handle: the size, the last write time GetFileTime reports (as nanoseconds since the
// Unix epoch), the volume serial number and 128-bit file id FileIdInfo reports (a hard link shares them);
// resolved_path is the final path GetFinalPathNameByHandleW gives, with a missing tail appended.
TEST(RegularFile, WindowsFileInfoFromHandle) {
  TempDir dir("win_file_info");
  const auto file = make_pattern_file(dir, 4321);
  HANDLE h = CreateFileW(file.c_str(), FILE_READ_ATTRIBUTES, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                         nullptr, OPEN_EXISTING, 0, nullptr);
  ASSERT_NE(h, INVALID_HANDLE_VALUE);
  BY_HANDLE_FILE_INFORMATION bh{};
  ASSERT_TRUE(GetFileInformationByHandle(h, &bh));
  FILE_ID_INFO fid{};
  ASSERT_TRUE(GetFileInformationByHandleEx(h, FileIdInfo, &fid, sizeof fid));
  std::wstring final_path(1024, L'\0');
  const DWORD n = GetFinalPathNameByHandleW(h, final_path.data(), static_cast<DWORD>(final_path.size()),
                                            FILE_NAME_NORMALIZED | VOLUME_NAME_DOS);
  CloseHandle(h);
  ASSERT_GT(n, 0u);
  final_path.resize(n);
  const mi::util::FileInfo info = mi::util::file_info(file);
  EXPECT_EQ(info.kind, mi::util::FileKind::kRegular);
  EXPECT_EQ(info.size, 4321u);
  EXPECT_EQ(info.volume, fid.VolumeSerialNumber);
  std::uint64_t id[2];
  std::memcpy(id, fid.FileId.Identifier, sizeof id);
  EXPECT_EQ(info.id, id[0]);
  EXPECT_EQ(info.id_high, id[1]);
  EXPECT_EQ(info.volume & 0xffffffffu, bh.dwVolumeSerialNumber);
  const auto ticks = static_cast<std::int64_t>((std::uint64_t{bh.ftLastWriteTime.dwHighDateTime} << 32) |
                                               bh.ftLastWriteTime.dwLowDateTime);
  const std::int64_t unix_ns = (ticks - 116444736000000000LL) * 100;  // FILETIME epoch 1601 -> 1970
  EXPECT_EQ(info.mtime_ns, unix_ns);
  ASSERT_TRUE(CreateHardLinkW((dir.path() / "link.bin").c_str(), file.c_str(), nullptr));
  EXPECT_TRUE(mi::util::same_file(mi::util::file_info(dir.path() / "link.bin"), info));
  EXPECT_EQ(mi::util::resolved_path(file), std::filesystem::path(final_path));
  EXPECT_EQ(mi::util::resolved_path(dir.path() / "new" / "out.bin"),
            mi::util::resolved_path(dir.path()) / "new" / "out.bin");
  EXPECT_EQ(mi::util::file_info(dir.path()).kind, mi::util::FileKind::kDirectory);
  EXPECT_EQ(mi::util::file_info(R"(\\.\pipe\x)").kind, mi::util::FileKind::kOther);  // refused by its text, unopened
}

// OutputFile::publish on Windows renames through the temporary's handle: the bytes appear at the target, in the
// target's folder (never in the current directory), and the temporary is gone; an existing target is replaced, also
// while another handle has it open with the engine readers' share mode (that reader keeps the old bytes); discard
// leaves nothing and the target as it was.
TEST(RegularFile, WindowsPublishRenamesInPlace) {
  namespace fs = std::filesystem;
  using Create = mi::util::OutputFile::Create;
  TempDir dir("win_publish");
  const fs::path target = dir.path() / "entry.bin";
  const auto publish = [&](std::string_view bytes) {
    mi::util::OutputFile out(target, Create::kTemporary);
    EXPECT_TRUE(out.is_open());
    const fs::path tmp = out.path();
    EXPECT_TRUE(out.write(bytes));
    const bool published = out.publish(target);
    EXPECT_FALSE(mi::util::file_info(tmp).exists()) << tmp.string();
    return published;
  };
  const auto contents = [&] {
    std::string text;
    EXPECT_EQ(mi::util::read_replaceable_file(target, 64, text), mi::util::ReadStatus::kOk);
    return text;
  };
  ASSERT_TRUE(publish("first"));
  EXPECT_EQ(contents(), "first");
  EXPECT_FALSE(mi::util::file_info(fs::current_path() / "entry.bin").exists());
  EXPECT_FALSE(mi::testing::temporary_left(target));
  ASSERT_TRUE(publish("second"));
  EXPECT_EQ(contents(), "second");
  {
    const FileReader reader(target);
    ASSERT_TRUE(publish("third"));
    char old[6] = {};
    reader.read_at(0, old, sizeof old);
    EXPECT_EQ(std::string(old, sizeof old), "second");
  }
  EXPECT_EQ(contents(), "third");
  mi::util::OutputFile dropped(target, Create::kTemporary);
  const fs::path dropped_path = dropped.path();
  EXPECT_TRUE(dropped.write("x", 1));
  dropped.discard();
  EXPECT_FALSE(mi::util::file_info(dropped_path).exists());
  EXPECT_EQ(contents(), "third");
  EXPECT_FALSE(mi::testing::temporary_left(target));
}
#endif

#ifdef __linux__
// A file under another process's lease (a write lease, broken by our open) is not waited on for the 45 s of the lease
// break: the non-blocking open is retried for about a second, then refused as leased.
TEST(FileReader, LeasedFileIsRefusedAfterRetries) {
  TempDir dir("fr_lease");
  const auto path = make_pattern_file(dir, 4096);
  int ready[2];
  ASSERT_EQ(::pipe(ready), 0);
  const pid_t child = ::fork();
  ASSERT_GE(child, 0);
  if (child == 0) {  // only async-signal-safe calls until _exit
    ::signal(SIGIO, SIG_IGN);  // the lease-break notice
    const int fd = ::open(path.c_str(), O_RDONLY);
    const char ok = fd >= 0 && ::fcntl(fd, F_SETLEASE, F_WRLCK) == 0 ? 'y' : 'n';
    if (::write(ready[1], &ok, 1) != 1) ::_exit(1);
    ::sleep(60);
    ::_exit(0);
  }
  ::close(ready[1]);
  char ok = 'n';
  const bool got = ::read(ready[0], &ok, 1) == 1;
  ::close(ready[0]);
  struct Reap {
    pid_t pid;
    ~Reap() {
      ::kill(pid, SIGKILL);
      ::waitpid(pid, nullptr, 0);
    }
  } reap{child};
  ASSERT_TRUE(got) << "the child did not report";
  // Leases need a file system that supports them (overlayfs before Linux 4.19 and some FUSE / network mounts do not).
  if (ok != 'y') GTEST_SKIP() << "this file system grants no write lease";
  const auto t0 = std::chrono::steady_clock::now();
  try {
    FileReader r(path);
    ADD_FAILURE() << "a leased file opened";
  } catch (const IoError& e) {
    EXPECT_EQ(std::string(e.what()), mi::util::path_utf8(path) + ": file is leased by another process (EWOULDBLOCK)");
  }
  const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  EXPECT_GT(s, 0.9);
  EXPECT_LT(s, 5.0);
}
#endif

TEST(MappedFile, MapsSafetensorsTensorBytes) {
  TempDir dir("mm_st");
  auto path = dir.path() / "a.safetensors";
  write_safetensors(path, {{"a", DType::kF32, {10}}, {"b", DType::kBF16, {33}}});
  SafetensorsFile f(path);
  MappedFile m(path);
  const TensorInfo* b = f.find("b");
  ASSERT_NE(b, nullptr);
  expect_pattern(m.data() + b->offset, b->offset - f.data_start(), b->nbytes);
}

}  // namespace
