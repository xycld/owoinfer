#include "util/xxh64.h"

#include <cstring>

namespace mi {

namespace {

constexpr std::uint64_t kPrime1 = 0x9E3779B185EBCA87ULL;
constexpr std::uint64_t kPrime2 = 0xC2B2AE3D27D4EB4FULL;
constexpr std::uint64_t kPrime3 = 0x165667B19E3779F9ULL;
constexpr std::uint64_t kPrime4 = 0x85EBCA77C2B2AE63ULL;
constexpr std::uint64_t kPrime5 = 0x27D4EB2F165667C5ULL;

constexpr std::uint64_t rotl(std::uint64_t x, int r) { return (x << r) | (x >> (64 - r)); }

// The format is defined little-endian; assemble bytes explicitly so big-endian hosts agree.
std::uint64_t read64(const std::uint8_t* p) {
  std::uint64_t v = 0;
  for (int i = 7; i >= 0; --i) v = (v << 8) | p[i];
  return v;
}
std::uint32_t read32(const std::uint8_t* p) {
  return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8) |
         (static_cast<std::uint32_t>(p[2]) << 16) | (static_cast<std::uint32_t>(p[3]) << 24);
}

std::uint64_t round(std::uint64_t acc, std::uint64_t input) {
  return rotl(acc + input * kPrime2, 31) * kPrime1;
}

std::uint64_t merge_round(std::uint64_t acc, std::uint64_t val) {
  return (acc ^ round(0, val)) * kPrime1 + kPrime4;
}

}  // namespace

Xxh64::Xxh64(std::uint64_t seed)
    : seed_(seed), acc_{seed + kPrime1 + kPrime2, seed + kPrime2, seed, seed - kPrime1}, buf_{} {}

void Xxh64::update(const void* data, std::size_t size) {
  const auto* p = static_cast<const std::uint8_t*>(data);
  total_ += size;
  if (buf_len_ + size < 32) {
    if (size != 0) std::memcpy(buf_ + buf_len_, p, size);
    buf_len_ += size;
    return;
  }
  if (buf_len_ != 0) {
    std::size_t fill = 32 - buf_len_;
    std::memcpy(buf_ + buf_len_, p, fill);
    for (int i = 0; i < 4; ++i) acc_[i] = round(acc_[i], read64(buf_ + 8 * i));
    p += fill;
    size -= fill;
    buf_len_ = 0;
  }
  while (size >= 32) {
    for (int i = 0; i < 4; ++i) acc_[i] = round(acc_[i], read64(p + 8 * i));
    p += 32;
    size -= 32;
  }
  if (size != 0) std::memcpy(buf_, p, size);
  buf_len_ = size;
}

std::uint64_t Xxh64::digest() const {
  std::uint64_t h;
  if (total_ >= 32) {
    h = rotl(acc_[0], 1) + rotl(acc_[1], 7) + rotl(acc_[2], 12) + rotl(acc_[3], 18);
    for (int i = 0; i < 4; ++i) h = merge_round(h, acc_[i]);
  } else {
    h = seed_ + kPrime5;
  }
  h += total_;

  const std::uint8_t* p = buf_;
  std::size_t left = buf_len_;
  while (left >= 8) {
    h ^= round(0, read64(p));
    h = rotl(h, 27) * kPrime1 + kPrime4;
    p += 8;
    left -= 8;
  }
  if (left >= 4) {
    h ^= static_cast<std::uint64_t>(read32(p)) * kPrime1;
    h = rotl(h, 23) * kPrime2 + kPrime3;
    p += 4;
    left -= 4;
  }
  while (left > 0) {
    h ^= *p * kPrime5;
    h = rotl(h, 11) * kPrime1;
    ++p;
    --left;
  }
  h ^= h >> 33;
  h *= kPrime2;
  h ^= h >> 29;
  h *= kPrime3;
  h ^= h >> 32;
  return h;
}

std::uint64_t xxh64(const void* data, std::size_t size, std::uint64_t seed) {
  Xxh64 h(seed);
  h.update(data, size);
  return h.digest();
}

}  // namespace mi
