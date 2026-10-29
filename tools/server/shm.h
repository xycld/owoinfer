// moss-server's shared-memory regions: the tensors of a component run and the scratch buffers of the VAE and
// conditioning calls travel through them instead of the JSON lines. Standalone (no engine, no CUDA runtime): the
// server pins a region's payload through a callback (the engine's, via the C API).
//
// Every region is created by the server, exclusively, readable and writable by the current user only:
//   Linux    shm_open("/mossinfer-<pid>-<seq>-<16 hex>", O_CREAT | O_EXCL | O_RDWR, 0600); a name that exists gets a
//            new random suffix. The name lives until unlink(): the server unlinks it once the client has attached (the
//            first request naming the region), after which a crash of either process leaks nothing. Regions of a
//            server that died before that are removed by the next server's sweep_stale(). The region's blocks are
//            reserved at creation (posix_fallocate), so a /dev/shm too small for it (a container's 64 MiB default) is an
//            error then, not a SIGBUS at a later first touch.
//   Windows  CreateFileMappingW over the paging file, "Local\mossinfer-<pid>-<seq>-<16 hex>", with a DACL granting the
//            current user's SID only; ERROR_ALREADY_EXISTS closes the handle and retries with a new name. A mapping
//            vanishes with its last handle.
// Layout (little-endian): a 64-byte header, then the payload at kPayloadOffset (page-aligned, for pinning):
//   0   magic "MOSSSHM\0"           8   u32 layout version (kLayoutVersion)   12  u32 server pid
//   16  token, 16 random bytes      32  u64 region bytes                     40  u64 payload offset
//   48  u64 the creating server's PID namespace (the inode of /proc/self/ns/pid; 0 on Windows or where unreadable)
//   56  reserved, zero
// A client checks the magic, the version and the token (returned by the creating request) before use, and that the
// region bytes lie between the payload offset and the size it mapped: on Windows a name recreated after its server died
// opens an empty mapping. The region's size is rounded up to the page size (Linux)
// or the allocation granularity (Windows) and recorded in the header.
// Teardown: a Region unpins, then unmaps, then unlinks. When the pin is the engine's (a run adopting the payload as its
// staging, which the run's end unregisters), the server ends the run before it destroys the region
// (docs/server-protocol.md, shared memory).
#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace mi::server::shm {

inline constexpr char kMagic[8] = {'M', 'O', 'S', 'S', 'S', 'H', 'M', '\0'};
inline constexpr std::uint32_t kLayoutVersion = 1;
inline constexpr std::size_t kHeaderBytes = 64;
inline constexpr std::size_t kPayloadOffset = 4096;
// Every region name starts with this (after "/" on Linux, "Local\" on Windows).
inline constexpr std::string_view kNamePrefix = "mossinfer-";

struct Token {
  std::array<std::uint8_t, 16> bytes{};
  std::string hex() const;  // 32 lowercase hex digits
  // From 32 hex digits (either case); false otherwise.
  static bool parse(std::string_view hex, Token& out);
  bool operator==(const Token&) const = default;
};

struct CreateOptions {
  // Pins the payload (e.g. the engine's cudaHostRegister through the C API); returns whether it did. Unset or false:
  // the region is used unpinned (the caller then copies through its own staging).
  std::function<bool(void* payload, std::size_t bytes)> pin;
  // Undoes a successful pin before the payload is unmapped (unset: nothing to undo).
  std::function<void(void* payload)> unpin;
  // Test seam: the name of creation attempt `attempt` (0, 1, ...; without the platform prefix); unset: kNamePrefix +
  // "<pid>-<seq>-<16 random hex>".
  std::function<std::string(int attempt)> name_for_tests;
  // Test seam (Linux): an error number to report instead of reserving the region's blocks (0: reserve them).
  std::function<int(std::size_t bytes)> allocate_for_tests;
};

class Region {
 public:
  // A new region whose payload holds `payload_bytes` (> 0), its blocks reserved, the header written and the pin
  // attempted. Throws std::system_error (the OS error; ENOSPC: "/dev/shm too small (N MiB needed ...)") or
  // std::invalid_argument.
  static Region create(std::size_t payload_bytes, const CreateOptions& options = {});
  Region() = default;
  ~Region();  // unpins, unmaps, unlinks (Linux) and closes the handle
  Region(Region&& other) noexcept;
  Region& operator=(Region&& other) noexcept;
  Region(const Region&) = delete;
  Region& operator=(const Region&) = delete;

  // As a client opens it: "/mossinfer-..." (Linux, the shm_open name) or "Local\mossinfer-..." (Windows, the tagname).
  const std::string& name() const { return name_; }
  const Token& token() const { return token_; }
  std::size_t bytes() const { return bytes_; }  // the whole mapping, header included
  std::byte* payload() const { return base_ == nullptr ? nullptr : base_ + kPayloadOffset; }
  std::size_t payload_bytes() const { return bytes_ - kPayloadOffset; }
  bool pinned() const { return pinned_; }
  int attempts() const { return attempts_; }  // creation attempts it took (> 1 after a name collision)
  // Linux: removes the name; the mapping stays valid in both processes. Idempotent; nothing on Windows.
  void unlink();
  bool linked() const { return linked_; }
  explicit operator bool() const { return base_ != nullptr; }

 private:
  void reset() noexcept;

  std::string name_;
  Token token_;
  std::byte* base_ = nullptr;
  std::size_t bytes_ = 0;
  bool pinned_ = false;
  bool linked_ = false;
  int attempts_ = 0;
  std::function<void(void*)> unpin_;
#ifdef _WIN32
  void* mapping_ = nullptr;  // HANDLE
#endif
};

// A client's mapping of a region (tests and C++ clients): opened by name, checked against the token.
class Attachment {
 public:
  // Throws std::system_error when the name cannot be opened, std::runtime_error when the header is not a region's or
  // the token differs.
  static Attachment open(const std::string& name, const Token& token);
  Attachment() = default;
  ~Attachment();
  Attachment(Attachment&& other) noexcept;
  Attachment& operator=(Attachment&& other) noexcept;
  Attachment(const Attachment&) = delete;
  Attachment& operator=(const Attachment&) = delete;

  std::byte* payload() const { return base_ == nullptr ? nullptr : base_ + kPayloadOffset; }
  std::size_t payload_bytes() const { return bytes_ - kPayloadOffset; }
  std::size_t region_bytes() const { return bytes_; }  // the header's region size
  std::size_t mapped_bytes() const { return mapped_; }  // what the client mapped (>= region_bytes())
  std::uint32_t server_pid() const { return pid_; }

 private:
  void reset() noexcept;

  std::byte* base_ = nullptr;
  std::size_t bytes_ = 0;   // from the header
  std::size_t mapped_ = 0;  // unmapped with this size
  std::uint32_t pid_ = 0;
#ifdef _WIN32
  void* mapping_ = nullptr;
#endif
};

// Linux: unlinks the regions a dead server left behind and returns their names: /dev/shm/<prefix><pid>-* owned by this
// user, <pid> not alive and not this process, and either a header naming that pid and this process's PID namespace
// (removed at once) or an unfinished creation (shorter than a header, or all zero) unchanged for at least `min_age`. A
// header of another pid or PID namespace is never removed: the namespace and age tests keep the sweep off the regions
// of a server in another PID namespace sharing /dev/shm (containers). A pid reused by a live process keeps its regions
// until that process exits. `prefix` is a test seam (tests sweep their own names only).
// Windows: none (mappings vanish with their processes); returns nothing.
std::vector<std::string> sweep_stale(std::string_view prefix = kNamePrefix,
                                     std::chrono::seconds min_age = std::chrono::minutes(10));

}  // namespace mi::server::shm
