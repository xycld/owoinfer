// tools/server/shm.h in one process: the header and the token check, exclusive creation through a name collision, the
// pin callback, unlink, and (Linux) the sweep of a dead server's regions. The cross-process cases (SIGKILL, Python
// clients) are tests/server/test_shm.py.
#include <gtest/gtest.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "../../tools/server/shm.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {

namespace shm = mi::server::shm;

std::uint32_t self_pid() {
#ifdef _WIN32
  return static_cast<std::uint32_t>(GetCurrentProcessId());
#else
  return static_cast<std::uint32_t>(getpid());
#endif
}

std::string bare(const std::string& name) {  // without "/" or "Local\"
  const std::size_t at = name.find(shm::kNamePrefix);
  return at == std::string::npos ? name : name.substr(at);
}

TEST(ServerShm, TokenHexRoundTrip) {
  shm::Token t;
  for (std::size_t i = 0; i < t.bytes.size(); ++i) t.bytes[i] = static_cast<std::uint8_t>(i * 17 + 1);
  const std::string hex = t.hex();
  EXPECT_EQ(hex.size(), 32u);
  EXPECT_EQ(hex.substr(0, 6), "011223");
  shm::Token back;
  ASSERT_TRUE(shm::Token::parse(hex, back));
  EXPECT_EQ(back, t);
  std::string upper = hex;
  for (char& c : upper) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  ASSERT_TRUE(shm::Token::parse(upper, back));
  EXPECT_EQ(back, t);
  EXPECT_FALSE(shm::Token::parse(hex.substr(1), back));
  EXPECT_FALSE(shm::Token::parse(hex.substr(0, 31) + "g", back));
}

TEST(ServerShm, CreateWritesTheHeaderAndAttachChecksTheToken) {
  shm::Region r = shm::Region::create(1000);
  ASSERT_TRUE(r);
  EXPECT_EQ(bare(r.name()).rfind("mossinfer-" + std::to_string(self_pid()) + "-", 0), 0u) << r.name();
  EXPECT_EQ(r.bytes() % 4096, 0u);
  EXPECT_GE(r.payload_bytes(), 1000u);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(r.payload()) % 4096, 0u);
  EXPECT_FALSE(r.pinned());
  EXPECT_EQ(r.attempts(), 1);
  const std::byte* h = r.payload() - shm::kPayloadOffset;
  EXPECT_EQ(std::memcmp(h, shm::kMagic, 8), 0);
  std::uint32_t version = 0, pid = 0;
  std::uint64_t bytes = 0, offset = 0;
  std::memcpy(&version, h + 8, 4);
  std::memcpy(&pid, h + 12, 4);
  std::memcpy(&bytes, h + 32, 8);
  std::memcpy(&offset, h + 40, 8);
  EXPECT_EQ(version, shm::kLayoutVersion);
  EXPECT_EQ(pid, self_pid());
  EXPECT_EQ(bytes, r.bytes());
  EXPECT_EQ(offset, shm::kPayloadOffset);
  EXPECT_EQ(std::memcmp(h + 16, r.token().bytes.data(), 16), 0);
#ifndef _WIN32
  struct stat ns {};
  ASSERT_EQ(stat("/proc/self/ns/pid", &ns), 0);
  std::uint64_t header_ns = 0;
  std::memcpy(&header_ns, h + 48, 8);
  EXPECT_EQ(header_ns, static_cast<std::uint64_t>(ns.st_ino));  // the creator's PID namespace
#endif
  for (std::size_t i = 56; i < shm::kHeaderBytes; ++i) EXPECT_EQ(h[i], std::byte{0}) << i;

  // A client's mapping sees the server's writes and vice versa.
  shm::Attachment a = shm::Attachment::open(r.name(), r.token());
  EXPECT_EQ(a.server_pid(), self_pid());
  EXPECT_EQ(a.payload_bytes(), r.payload_bytes());
  EXPECT_EQ(a.region_bytes(), r.bytes());
  EXPECT_GE(a.mapped_bytes(), a.region_bytes());
  r.payload()[5] = std::byte{42};
  EXPECT_EQ(a.payload()[5], std::byte{42});
  a.payload()[6] = std::byte{7};
  EXPECT_EQ(r.payload()[6], std::byte{7});

  shm::Token wrong = r.token();
  wrong.bytes[15] ^= 1;
  try {
    shm::Attachment::open(r.name(), wrong);
    FAIL();
  } catch (const std::runtime_error& e) {
    EXPECT_NE(std::string(e.what()).find("another token"), std::string::npos) << e.what();
  }
  EXPECT_THROW(shm::Region::create(0), std::invalid_argument);
}

TEST(ServerShm, ACollisionRetriesWithANewName) {
  shm::Region a = shm::Region::create(4096);
  const std::string taken = bare(a.name());
  shm::CreateOptions o;
  o.name_for_tests = [&](int attempt) { return attempt == 0 ? taken : taken + "-" + std::to_string(attempt); };
  shm::Region b = shm::Region::create(4096, o);
  EXPECT_EQ(b.attempts(), 2);
  EXPECT_EQ(bare(b.name()), taken + "-1");
  // The first region is untouched by the attempt.
  EXPECT_NO_THROW(shm::Attachment::open(a.name(), a.token()));
  EXPECT_NO_THROW(shm::Attachment::open(b.name(), b.token()));
  shm::CreateOptions always;
  always.name_for_tests = [&](int) { return taken; };
  EXPECT_THROW(shm::Region::create(4096, always), std::runtime_error);
}

TEST(ServerShm, PinCallbackAndTeardownOrder) {
  void* pinned_at = nullptr;
  std::size_t pinned_bytes = 0;
  void* unpinned_at = nullptr;
  {
    shm::CreateOptions o;
    o.pin = [&](void* p, std::size_t n) {
      pinned_at = p;
      pinned_bytes = n;
      return true;
    };
    o.unpin = [&](void* p) { unpinned_at = p; };
    shm::Region r = shm::Region::create(10000, o);
    EXPECT_TRUE(r.pinned());
    EXPECT_EQ(pinned_at, r.payload());
    EXPECT_EQ(pinned_bytes, r.payload_bytes());
    shm::Region moved = std::move(r);  // the move carries the pin; the moved-from region undoes nothing
    EXPECT_FALSE(r);
    EXPECT_EQ(unpinned_at, nullptr);
  }
  EXPECT_EQ(unpinned_at, pinned_at);
  // A refused pin: the region is used unpinned and nothing is undone.
  bool unpin_called = false;
  {
    shm::CreateOptions o;
    o.pin = [](void*, std::size_t) { return false; };
    o.unpin = [&](void*) { unpin_called = true; };
    shm::Region r = shm::Region::create(4096, o);
    EXPECT_FALSE(r.pinned());
  }
  EXPECT_FALSE(unpin_called);
}

#ifndef _WIN32
TEST(ServerShm, UnlinkRemovesTheNameNotTheMapping) {
  shm::Region r = shm::Region::create(4096);
  shm::Attachment a = shm::Attachment::open(r.name(), r.token());
  EXPECT_TRUE(r.linked());
  r.unlink();
  r.unlink();  // idempotent
  EXPECT_FALSE(r.linked());
  EXPECT_THROW(shm::Attachment::open(r.name(), r.token()), std::system_error);
  r.payload()[0] = std::byte{9};
  EXPECT_EQ(a.payload()[0], std::byte{9});
  struct stat st {};
  EXPECT_NE(stat(("/dev/shm" + r.name()).c_str(), &st), 0);
}

TEST(ServerShm, RegionsAreReadableByTheUserOnly) {
  shm::Region r = shm::Region::create(4096);
  struct stat st {};
  ASSERT_EQ(stat(("/dev/shm" + r.name()).c_str(), &st), 0);
  EXPECT_EQ(st.st_mode & 0777, 0600u);
  EXPECT_EQ(st.st_uid, geteuid());
  EXPECT_EQ(static_cast<std::size_t>(st.st_size), r.bytes());
}

// A pid that is certainly dead: a child that exited and was reaped.
pid_t dead_pid() {
  const pid_t child = fork();
  if (child == 0) _exit(0);
  int status = 0;
  waitpid(child, &status, 0);
  return child;
}

std::uint64_t own_pid_namespace() {
  struct stat st {};
  return stat("/proc/self/ns/pid", &st) == 0 ? static_cast<std::uint64_t>(st.st_ino) : 0;
}

// An shm object with `size` bytes whose header names `header_pid` and PID namespace `ns` (0 bytes: no header).
void make_object(const std::string& name, std::size_t size, std::uint32_t header_pid, std::uint64_t ns = own_pid_namespace()) {
  const int fd = shm_open(name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
  ASSERT_GE(fd, 0) << name;
  if (size > 0) {
    ASSERT_EQ(ftruncate(fd, static_cast<off_t>(size)), 0);
    std::byte header[shm::kHeaderBytes] = {};
    std::memcpy(header, shm::kMagic, 8);
    std::memcpy(header + 12, &header_pid, 4);
    std::memcpy(header + 48, &ns, 8);
    ASSERT_EQ(pwrite(fd, header, sizeof(header), 0), static_cast<ssize_t>(sizeof(header)));
  }
  close(fd);
}

bool exists(const std::string& name) {
  struct stat st {};
  return stat(("/dev/shm" + name).c_str(), &st) == 0;
}

TEST(ServerShm, SweepRemovesOnlyADeadServersRegions) {
  ASSERT_NE(own_pid_namespace(), 0u) << "/proc/self/ns/pid unreadable";
  // This test's own prefix: a concurrent test's (or server's) regions are never looked at.
  const std::string prefix = "mossinfer-ut" + std::to_string(getpid()) + "-";
  const pid_t dead = dead_pid();
  const std::string at = "/" + prefix + std::to_string(dead);
  const std::string stale = at + "-0", unfinished = at + "-1", foreign = at + "-2", other_ns = at + "-3";
  const std::string odd = "/" + prefix + "x" + std::to_string(dead) + "-4";
  for (const std::string& n : {stale, unfinished, foreign, other_ns, odd}) shm_unlink(n.c_str());
  make_object(stale, 8192, static_cast<std::uint32_t>(dead));
  make_object(unfinished, 0, 0);                                                  // killed before ftruncate
  make_object(foreign, 8192, static_cast<std::uint32_t>(dead) + 1);                // its header names another pid
  make_object(other_ns, 8192, static_cast<std::uint32_t>(dead), own_pid_namespace() + 1);  // another PID namespace
  make_object(odd, 8192, static_cast<std::uint32_t>(dead));                        // not <prefix><digits>-
  shm::CreateOptions o;
  o.name_for_tests = [&](int attempt) { return prefix + std::to_string(getpid()) + "-live-" + std::to_string(attempt); };
  shm::Region live = shm::Region::create(4096, o);  // this process: alive
  // A server's sweep (the 10-minute floor): the proven region of the dead pid goes at once, the seconds-old unfinished
  // creation stays (it may be another PID namespace's, under way).
  const std::vector<std::string> first = shm::sweep_stale(prefix);
  EXPECT_EQ(first, std::vector<std::string>{stale});
  EXPECT_FALSE(exists(stale));
  EXPECT_TRUE(exists(unfinished));
  // With no age floor the unfinished creation goes too; the rest is never removed.
  const std::vector<std::string> removed = shm::sweep_stale(prefix, std::chrono::seconds(0));
  EXPECT_EQ(removed, std::vector<std::string>{unfinished});
  EXPECT_FALSE(exists(unfinished));
  EXPECT_TRUE(exists(foreign));
  EXPECT_TRUE(exists(other_ns));
  EXPECT_TRUE(exists(live.name()));
  for (const std::string& n : {foreign, other_ns, odd}) shm_unlink(n.c_str());
}

TEST(ServerShm, ATooSmallDevShmIsAnErrorNotASigbus) {
  // The reservation failing as on a full tmpfs: ENOSPC with the size needed, and no name left behind.
  const std::string name = "mossinfer-ut" + std::to_string(getpid()) + "-nospace";
  shm::CreateOptions o;
  o.name_for_tests = [&](int) { return name; };
  std::size_t asked = 0;
  o.allocate_for_tests = [&](std::size_t bytes) {
    asked = bytes;
    return ENOSPC;
  };
  try {
    shm::Region::create(3 * 1024 * 1024, o);
    FAIL() << "expected ENOSPC";
  } catch (const std::system_error& e) {
    EXPECT_EQ(e.code().value(), ENOSPC);
    EXPECT_NE(std::string(e.what()).find("/dev/shm too small (4 MiB needed"), std::string::npos) << e.what();
  }
  EXPECT_GE(asked, std::size_t{3 * 1024 * 1024 + 4096});
  EXPECT_FALSE(exists("/" + name));
  // Another reservation error keeps its own words.
  o.allocate_for_tests = [](std::size_t) { return EIO; };
  try {
    shm::Region::create(4096, o);
    FAIL() << "expected EIO";
  } catch (const std::system_error& e) {
    EXPECT_EQ(e.code().value(), EIO);
    EXPECT_NE(std::string(e.what()).find("posix_fallocate"), std::string::npos) << e.what();
  }
  EXPECT_FALSE(exists("/" + name));
  // Without the seam the blocks are reserved: the whole region is allocated on tmpfs, not sparse.
  shm::CreateOptions plain;
  plain.name_for_tests = [&](int) { return name; };
  shm::Region r = shm::Region::create(2 * 1024 * 1024, plain);
  struct stat st {};
  ASSERT_EQ(stat(("/dev/shm/" + name).c_str(), &st), 0);
  EXPECT_GE(static_cast<std::size_t>(st.st_blocks) * 512, r.bytes());
}
#endif

}  // namespace
