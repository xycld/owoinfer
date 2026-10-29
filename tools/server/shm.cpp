#include "shm.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <ctime>
#include <stdexcept>
#include <system_error>
#include <utility>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
// windows.h first: bcrypt.h needs its types.
#include <bcrypt.h>
#ifdef _MSC_VER
#pragma comment(lib, "bcrypt.lib")
#endif
#else
#include <dirent.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/random.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace mi::server::shm {
namespace {

constexpr int kMaxAttempts = 16;
constexpr char kHexDigits[] = "0123456789abcdef";

std::string to_hex(const std::uint8_t* p, std::size_t n) {
  std::string s(2 * n, '0');
  for (std::size_t i = 0; i < n; ++i) {
    s[2 * i] = kHexDigits[p[i] >> 4];
    s[2 * i + 1] = kHexDigits[p[i] & 15];
  }
  return s;
}

#ifdef _WIN32
// `err`: the GetLastError() value captured right after the failing call (a destructor in between may change it).
[[noreturn]] void throw_win(DWORD err, const std::string& what) {
  throw std::system_error(static_cast<int>(err), std::system_category(), what);
}
[[noreturn]] void throw_os(const std::string& what) { throw_win(GetLastError(), what); }
#else
[[noreturn]] void throw_os(const std::string& what) { throw std::system_error(errno, std::generic_category(), what); }
#endif

// The OS's cryptographic random source.
void random_bytes(std::uint8_t* out, std::size_t n) {
#ifdef _WIN32
  if (BCryptGenRandom(nullptr, out, static_cast<ULONG>(n), BCRYPT_USE_SYSTEM_PREFERRED_RNG) != 0) {
    throw std::runtime_error("shm: BCryptGenRandom failed");
  }
#else
  while (n > 0) {
    const ssize_t got = getrandom(out, n, 0);
    if (got < 0) {
      if (errno == EINTR) continue;
      throw_os("shm: getrandom");
    }
    out += got;
    n -= static_cast<std::size_t>(got);
  }
#endif
}

std::uint32_t current_pid() {
#ifdef _WIN32
  return static_cast<std::uint32_t>(GetCurrentProcessId());
#else
  return static_cast<std::uint32_t>(getpid());
#endif
}

std::string default_name() {
  static std::atomic<std::uint64_t> seq{0};
  std::uint8_t suffix[8];
  random_bytes(suffix, sizeof(suffix));
  return std::string(kNamePrefix) + std::to_string(current_pid()) + "-" + std::to_string(seq.fetch_add(1)) + "-" +
         to_hex(suffix, sizeof(suffix));
}

void put_u32(std::byte* p, std::uint32_t v) {
  for (int i = 0; i < 4; ++i) p[i] = static_cast<std::byte>(v >> (8 * i));
}
void put_u64(std::byte* p, std::uint64_t v) {
  for (int i = 0; i < 8; ++i) p[i] = static_cast<std::byte>(v >> (8 * i));
}
std::uint32_t get_u32(const std::byte* p) {
  std::uint32_t v = 0;
  for (int i = 0; i < 4; ++i) v |= static_cast<std::uint32_t>(p[i]) << (8 * i);
  return v;
}
std::uint64_t get_u64(const std::byte* p) {
  std::uint64_t v = 0;
  for (int i = 0; i < 8; ++i) v |= static_cast<std::uint64_t>(p[i]) << (8 * i);
  return v;
}

// The inode of this process's PID namespace (Linux; 0 where it cannot be read, and on Windows).
std::uint64_t pid_namespace() {
#ifdef _WIN32
  return 0;
#else
  struct stat st {};
  return stat("/proc/self/ns/pid", &st) == 0 ? static_cast<std::uint64_t>(st.st_ino) : 0;
#endif
}

void write_header(std::byte* base, const Token& token, std::size_t bytes) {
  std::memset(base, 0, kHeaderBytes);
  std::memcpy(base, kMagic, sizeof(kMagic));
  put_u32(base + 8, kLayoutVersion);
  put_u32(base + 12, current_pid());
  std::memcpy(base + 16, token.bytes.data(), token.bytes.size());
  put_u64(base + 32, bytes);
  put_u64(base + 40, kPayloadOffset);
  put_u64(base + 48, pid_namespace());
}

// The header's check, as a client makes it; returns the server pid.
std::uint32_t check_header(const std::byte* base, std::size_t mapped, const Token& token, const std::string& name) {
  const std::uint64_t region = mapped >= kHeaderBytes ? get_u64(base + 32) : 0;
  if (mapped < kPayloadOffset || std::memcmp(base, kMagic, sizeof(kMagic)) != 0 || get_u32(base + 8) != kLayoutVersion ||
      get_u64(base + 40) != kPayloadOffset || region < kPayloadOffset || region > mapped) {
    throw std::runtime_error("shm: " + name + " is not a moss-server region of layout version " +
                             std::to_string(kLayoutVersion));
  }
  if (std::memcmp(base + 16, token.bytes.data(), token.bytes.size()) != 0) {
    throw std::runtime_error("shm: " + name + " has another token (a different region under the same name)");
  }
  return get_u32(base + 12);
}

std::size_t round_up(std::size_t n, std::size_t a) { return (n + a - 1) / a * a; }

// What a region's size is rounded to: the page size (Linux) or the allocation granularity (Windows), and never less
// than the payload offset.
std::size_t granularity() {
#ifdef _WIN32
  SYSTEM_INFO info{};
  GetSystemInfo(&info);
  const std::size_t g = info.dwAllocationGranularity;
#else
  const long page = sysconf(_SC_PAGESIZE);
  const std::size_t g = page > 0 ? static_cast<std::size_t>(page) : kPayloadOffset;
#endif
  return g > kPayloadOffset ? g : kPayloadOffset;
}

#ifdef _WIN32
std::wstring widen(const std::string& s) { return std::wstring(s.begin(), s.end()); }  // names are ASCII

// A security descriptor whose DACL grants the current user (the process token's user SID) full access to the mapping,
// and nobody else.
class UserOnlySecurity {
 public:
  UserOnlySecurity() {
    HANDLE token = nullptr;
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &token)) throw_os("shm: OpenProcessToken");
    DWORD len = 0;
    GetTokenInformation(token, TokenUser, nullptr, 0, &len);
    user_.resize(len);
    const BOOL ok = GetTokenInformation(token, TokenUser, user_.data(), len, &len);
    CloseHandle(token);
    if (!ok) throw_os("shm: GetTokenInformation");
    const PSID sid = reinterpret_cast<TOKEN_USER*>(user_.data())->User.Sid;
    const DWORD acl_len = static_cast<DWORD>(sizeof(ACL) + sizeof(ACCESS_ALLOWED_ACE) - sizeof(DWORD) + GetLengthSid(sid));
    acl_.resize(acl_len);
    auto* acl = reinterpret_cast<PACL>(acl_.data());
    if (!InitializeAcl(acl, acl_len, ACL_REVISION) || !AddAccessAllowedAce(acl, ACL_REVISION, FILE_MAP_ALL_ACCESS, sid) ||
        !InitializeSecurityDescriptor(&sd_, SECURITY_DESCRIPTOR_REVISION) || !SetSecurityDescriptorDacl(&sd_, TRUE, acl, FALSE)) {
      throw_os("shm: building the region's DACL");
    }
    sa_.nLength = sizeof(sa_);
    sa_.lpSecurityDescriptor = &sd_;
    sa_.bInheritHandle = FALSE;
  }
  SECURITY_ATTRIBUTES* attributes() { return &sa_; }

 private:
  std::vector<std::uint8_t> user_, acl_;
  SECURITY_DESCRIPTOR sd_{};
  SECURITY_ATTRIBUTES sa_{};
};
#endif

}  // namespace

std::string Token::hex() const { return to_hex(bytes.data(), bytes.size()); }

bool Token::parse(std::string_view hex, Token& out) {
  if (hex.size() != 2 * out.bytes.size()) return false;
  const auto nibble = [](char c) -> int {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
  };
  Token t;
  for (std::size_t i = 0; i < t.bytes.size(); ++i) {
    const int hi = nibble(hex[2 * i]), lo = nibble(hex[2 * i + 1]);
    if (hi < 0 || lo < 0) return false;
    t.bytes[i] = static_cast<std::uint8_t>(hi * 16 + lo);
  }
  out = t;
  return true;
}

Region Region::create(std::size_t payload_bytes, const CreateOptions& options) {
  if (payload_bytes == 0) throw std::invalid_argument("shm: a region needs a payload");
  if (payload_bytes > (std::size_t{1} << 48)) throw std::invalid_argument("shm: region too large");
  const std::size_t bytes = round_up(kPayloadOffset + payload_bytes, granularity());
  Region r;
  random_bytes(r.token_.bytes.data(), r.token_.bytes.size());
#ifdef _WIN32
  UserOnlySecurity security;
  for (int attempt = 0; attempt < kMaxAttempts && r.mapping_ == nullptr; ++attempt) {
    const std::string name = "Local\\" + (options.name_for_tests ? options.name_for_tests(attempt) : default_name());
    const std::wstring wname = widen(name);  // bound first: no destructor between the call and GetLastError
    r.attempts_ = attempt + 1;
    HANDLE h = CreateFileMappingW(INVALID_HANDLE_VALUE, security.attributes(), PAGE_READWRITE,
                                  static_cast<DWORD>(static_cast<std::uint64_t>(bytes) >> 32),
                                  static_cast<DWORD>(bytes & 0xffffffffu), wname.c_str());
    const DWORD err = GetLastError();
    if (h != nullptr && err == ERROR_ALREADY_EXISTS) {
      CloseHandle(h);  // somebody's mapping: not ours to use
      continue;
    }
    if (h == nullptr) {
      if (err == ERROR_ACCESS_DENIED) continue;  // the name exists with a DACL that is not ours
      throw_win(err, "shm: CreateFileMappingW " + name);
    }
    r.mapping_ = h;
    r.name_ = name;
  }
  if (r.mapping_ == nullptr) throw std::runtime_error("shm: no free region name after " + std::to_string(kMaxAttempts) + " attempts");
  void* p = MapViewOfFile(static_cast<HANDLE>(r.mapping_), FILE_MAP_ALL_ACCESS, 0, 0, bytes);
  if (p == nullptr) throw_os("shm: MapViewOfFile " + r.name_);  // ~Region closes the handle
  r.base_ = static_cast<std::byte*>(p);
#else
  int fd = -1;
  for (int attempt = 0; attempt < kMaxAttempts && fd < 0; ++attempt) {
    const std::string name = "/" + (options.name_for_tests ? options.name_for_tests(attempt) : default_name());
    r.attempts_ = attempt + 1;
    fd = shm_open(name.c_str(), O_CREAT | O_EXCL | O_RDWR | O_CLOEXEC, 0600);
    if (fd >= 0) {
      r.name_ = name;
      r.linked_ = true;
    } else if (errno != EEXIST) {
      throw_os("shm: shm_open " + name);
    }
  }
  if (fd < 0) throw std::runtime_error("shm: no free region name after " + std::to_string(kMaxAttempts) + " attempts");
  if (ftruncate(fd, static_cast<off_t>(bytes)) != 0) {
    const int err = errno;
    close(fd);
    errno = err;
    throw_os("shm: ftruncate " + r.name_);  // ~Region unlinks the name
  }
  // tmpfs is sparse: without its blocks reserved now, a first touch past /dev/shm's limit (a container's 64 MiB) would be
  // a SIGBUS in either process instead of an error here. posix_fallocate returns its error (errno is untouched).
  int alloc = options.allocate_for_tests ? options.allocate_for_tests(bytes) : 0;
  if (alloc == 0) {
    do {
      alloc = posix_fallocate(fd, 0, static_cast<off_t>(bytes));
    } while (alloc == EINTR);
  }
  if (alloc != 0) {
    close(fd);
    const std::string mib = std::to_string((bytes + (std::size_t{1} << 20) - 1) >> 20) + " MiB";
    throw std::system_error(alloc, std::generic_category(),
                            alloc == ENOSPC ? "shm: /dev/shm too small (" + mib + " needed for " + r.name_ + ")"
                                            : "shm: posix_fallocate " + r.name_ + " (" + mib + ")");
  }
  void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  const int err = errno;
  close(fd);  // the mapping keeps the object
  if (p == MAP_FAILED) {
    errno = err;
    throw_os("shm: mmap " + r.name_);
  }
  r.base_ = static_cast<std::byte*>(p);
#endif
  r.bytes_ = bytes;
  write_header(r.base_, r.token_, bytes);
  if (options.pin) {
    r.pinned_ = options.pin(r.payload(), r.payload_bytes());
    if (r.pinned_) r.unpin_ = options.unpin;
  }
  return r;
}

Region::~Region() { reset(); }

Region::Region(Region&& other) noexcept { *this = std::move(other); }

Region& Region::operator=(Region&& other) noexcept {
  if (this != &other) {
    reset();
    name_ = std::move(other.name_);
    token_ = other.token_;
    base_ = std::exchange(other.base_, nullptr);
    bytes_ = std::exchange(other.bytes_, 0);
    pinned_ = std::exchange(other.pinned_, false);
    linked_ = std::exchange(other.linked_, false);
    attempts_ = std::exchange(other.attempts_, 0);
    unpin_ = std::move(other.unpin_);
    other.unpin_ = nullptr;
#ifdef _WIN32
    mapping_ = std::exchange(other.mapping_, nullptr);
#endif
  }
  return *this;
}

void Region::unlink() {
#ifndef _WIN32
  if (linked_) shm_unlink(name_.c_str());  // ENOENT (already gone) is fine
#endif
  linked_ = false;
}

void Region::reset() noexcept {
  if (pinned_ && unpin_ && base_ != nullptr) unpin_(payload());
  pinned_ = false;
  unpin_ = nullptr;
#ifdef _WIN32
  if (base_ != nullptr) UnmapViewOfFile(base_);
  if (mapping_ != nullptr) CloseHandle(static_cast<HANDLE>(mapping_));
  mapping_ = nullptr;
#else
  if (base_ != nullptr) munmap(base_, bytes_);
  if (linked_) shm_unlink(name_.c_str());
#endif
  linked_ = false;
  base_ = nullptr;
  bytes_ = 0;
}

Attachment Attachment::open(const std::string& name, const Token& token) {
  Attachment a;
#ifdef _WIN32
  const std::wstring wname = widen(name);
  a.mapping_ = OpenFileMappingW(FILE_MAP_ALL_ACCESS, FALSE, wname.c_str());
  const DWORD err = GetLastError();
  if (a.mapping_ == nullptr) throw_win(err, "shm: OpenFileMappingW " + name);
  void* p = MapViewOfFile(static_cast<HANDLE>(a.mapping_), FILE_MAP_ALL_ACCESS, 0, 0, 0);
  if (p == nullptr) throw_os("shm: MapViewOfFile " + name);
  a.base_ = static_cast<std::byte*>(p);
  MEMORY_BASIC_INFORMATION info{};
  if (VirtualQuery(p, &info, sizeof(info)) == 0) throw_os("shm: VirtualQuery " + name);
  a.mapped_ = info.RegionSize;
#else
  const int fd = shm_open(name.c_str(), O_RDWR | O_CLOEXEC, 0);
  if (fd < 0) throw_os("shm: shm_open " + name);
  struct stat st {};
  if (fstat(fd, &st) != 0) {
    const int err = errno;
    close(fd);
    errno = err;
    throw_os("shm: fstat " + name);
  }
  const auto size = static_cast<std::size_t>(st.st_size);
  void* p = size == 0 ? MAP_FAILED : mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);
  if (p == MAP_FAILED) throw std::runtime_error("shm: cannot map " + name + " (" + std::to_string(size) + " bytes)");
  a.base_ = static_cast<std::byte*>(p);
  a.mapped_ = size;
#endif
  a.pid_ = check_header(a.base_, a.mapped_, token, name);  // a throw unmaps through ~Attachment
  a.bytes_ = static_cast<std::size_t>(get_u64(a.base_ + 32));
  return a;
}

Attachment::~Attachment() { reset(); }

Attachment::Attachment(Attachment&& other) noexcept { *this = std::move(other); }

Attachment& Attachment::operator=(Attachment&& other) noexcept {
  if (this != &other) {
    reset();
    base_ = std::exchange(other.base_, nullptr);
    bytes_ = std::exchange(other.bytes_, 0);
    mapped_ = std::exchange(other.mapped_, 0);
    pid_ = std::exchange(other.pid_, 0);
#ifdef _WIN32
    mapping_ = std::exchange(other.mapping_, nullptr);
#endif
  }
  return *this;
}

void Attachment::reset() noexcept {
#ifdef _WIN32
  if (base_ != nullptr) UnmapViewOfFile(base_);
  if (mapping_ != nullptr) CloseHandle(static_cast<HANDLE>(mapping_));
  mapping_ = nullptr;
#else
  if (base_ != nullptr) munmap(base_, mapped_);
#endif
  base_ = nullptr;
  bytes_ = 0;
  mapped_ = 0;
}

std::vector<std::string> sweep_stale(std::string_view prefix, std::chrono::seconds min_age) {
  std::vector<std::string> removed;
#ifdef _WIN32
  (void)prefix;
  (void)min_age;
#else
  DIR* dir = opendir("/dev/shm");
  if (dir == nullptr) return removed;
  const pid_t self = getpid();
  const uid_t uid = geteuid();
  const std::uint64_t ns = pid_namespace();
  const std::time_t now = std::time(nullptr);
  while (const dirent* e = readdir(dir)) {
    const std::string_view entry(e->d_name);
    if (prefix.empty() || entry.substr(0, prefix.size()) != prefix) continue;
    // <prefix><pid>-...
    std::size_t i = prefix.size();
    long long pid = 0;
    while (i < entry.size() && entry[i] >= '0' && entry[i] <= '9' && pid < (1LL << 31)) pid = pid * 10 + (entry[i++] - '0');
    if (i == prefix.size() || i >= entry.size() || entry[i] != '-' || pid <= 0 || pid >= (1LL << 31)) continue;
    struct stat st {};
    if (fstatat(dirfd(dir), e->d_name, &st, AT_SYMLINK_NOFOLLOW) != 0 || !S_ISREG(st.st_mode) || st.st_uid != uid) continue;
    // Young objects are left alone: a server of another PID namespace sharing /dev/shm (a container) may be creating or
    // using one under a pid that means nothing here.
    if (now - std::max(st.st_mtime, st.st_ctime) < static_cast<std::time_t>(min_age.count())) continue;
    if (static_cast<pid_t>(pid) == self) continue;
    if (kill(static_cast<pid_t>(pid), 0) == 0 || errno != ESRCH) continue;  // alive (EPERM: another user's live pid)
    // A complete header must be a region's of that pid created in this PID namespace (another namespace's pids are not
    // ours to test); a shorter or all-zero file is a creation its server did not finish (killed before ftruncate or the
    // header write).
    if (st.st_size >= static_cast<off_t>(kHeaderBytes)) {
      const int fd = openat(dirfd(dir), e->d_name, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
      if (fd < 0) continue;
      std::byte header[kHeaderBytes];
      const ssize_t got = pread(fd, header, sizeof(header), 0);
      close(fd);
      const bool zero = got == static_cast<ssize_t>(sizeof(header)) &&
                        std::all_of(header, header + sizeof(header), [](std::byte b) { return b == std::byte{0}; });
      if (!zero && (got != static_cast<ssize_t>(sizeof(header)) || std::memcmp(header, kMagic, sizeof(kMagic)) != 0 ||
                    get_u32(header + 12) != static_cast<std::uint32_t>(pid) || ns == 0 || get_u64(header + 48) != ns)) {
        continue;
      }
    }
    const std::string name = "/" + std::string(entry);
    if (shm_unlink(name.c_str()) == 0) removed.push_back(name);
  }
  closedir(dir);
#endif
  return removed;
}

}  // namespace mi::server::shm
