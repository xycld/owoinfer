// XXH64 (64-bit xxHash), implemented from the public algorithm specification
// (https://github.com/Cyan4973/xxHash/blob/dev/doc/xxhash_spec.md). The algorithm and its
// constants are BSD-2-Clause material from Yann Collet's xxHash; this file is an independent
// implementation. Used for non-cryptographic integrity checksums of weight groups.
#pragma once

#include <cstddef>
#include <cstdint>

namespace mi {

// Streaming hasher; feeding the same bytes in any chunking yields the same digest.
class Xxh64 {
 public:
  explicit Xxh64(std::uint64_t seed = 0);
  void update(const void* data, std::size_t size);
  std::uint64_t digest() const;

 private:
  std::uint64_t seed_;
  std::uint64_t acc_[4];
  std::uint8_t buf_[32];
  std::size_t buf_len_ = 0;
  std::uint64_t total_ = 0;
};

std::uint64_t xxh64(const void* data, std::size_t size, std::uint64_t seed = 0);

}  // namespace mi
