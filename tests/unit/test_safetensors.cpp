#include <gtest/gtest.h>

#include <atomic>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <numeric>
#include <string>
#include <thread>

#ifdef __linux__
#include <fcntl.h>
#include <sys/vfs.h>
#include <unistd.h>
#endif
#ifndef _WIN32
#include <sys/stat.h>

#include <cerrno>
#endif

#include "test_util.h"
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
// read_replaceable_file (certificates, caches). error_within fails the test instead of hanging on a regression.
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
